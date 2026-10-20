"""Local numerical/API checks. These do not certify Base checkpoint behavior."""

import argparse
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


parser = argparse.ArgumentParser()
parser.add_argument("--comfy-root", required=True)
parser.add_argument("--reference-root", required=True)
parser.add_argument("--report", required=True)
args = parser.parse_args()
sys.argv = [sys.argv[0], "--cpu"]
sys.path.insert(0, args.comfy_root)
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("moss_single_edit_under_test", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)

import torch
import torch.nn.functional as F
import comfy.model_management
import comfy.ops
import comfy.samplers
from comfy.ldm.moss_image.model import MOSSImage
from comfy.text_encoders.qwen_vl import process_qwen2vl_images
from transformers import AutoImageProcessor
from moss_single_edit_under_test import reference_images, model_patch, sampling_compat, loaders


METRICS = {}


def reference_function(filename, name, namespace):
    source = (Path(args.reference_root) / filename).read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
    node.decorator_list = []
    code = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(code)
    exec(compile(code, str(Path(args.reference_root) / filename), "exec"), namespace)
    return namespace[name]


class ContractChecks(unittest.TestCase):
    def test_sigma_grid_matches_training_source(self):
        reference = reference_function("pipeline.py", "moss_sigmas", {"torch": torch, "DEFAULT_SHIFT": 5.0, "NUM_TRAIN_TIMESTEPS": 1000})
        for steps, shift in [(1, 5), (8, 5), (50, 5), (1200, 5), (50, 3)]:
            expected = reference(steps, shift)
            actual = sampling_compat.moss_sigmas(steps, shift)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        METRICS["sigmas_50_shift_5"] = sampling_compat.moss_sigmas(50).tolist()

    def test_detail_resize_matches_training_source(self):
        import math
        reference = reference_function("moss_image.py", "_encode_ref_latents", {"torch": torch, "F": F, "math": math, "Optional": __import__('typing').Optional, "List": __import__('typing').List})
        captured = []
        fake = SimpleNamespace(device_torch=torch.device("cpu"), torch_dtype=torch.float32, model_config=SimpleNamespace(model_kwargs={"match_target_res": True}), get_bucket_divisibility=lambda: 16, _ref_target_pixels=lambda pixels: pixels, encode_images=lambda image, **kw: captured.append(image.clone()) or torch.zeros(1, 128, image.shape[-2] // 16, image.shape[-1] // 16))
        for shape, target in [((1, 3, 137, 229), (1024, 1024)), ((1, 3, 61, 203), (880, 1184)), ((1, 3, 128, 128), (512, 1024))]:
            source = torch.rand(shape)
            reference(fake, [source], target_pixels=target[0] * target[1])
            actual = reference_images.resize_reference(source, *target)
            torch.testing.assert_close(actual * 2 - 1, captured[-1], rtol=0, atol=0)
        self.assertNotEqual(actual.shape[-2:], (target[1], target[0]))

    def test_vision_processor_matches_local_transformers(self):
        import math
        ref = reference_function("moss_image.py", "_prep_vlm_images", {"torch": torch, "F": F, "math": math, "List": __import__('typing').List})
        fake = SimpleNamespace(device_torch=torch.device("cpu"), model_config=SimpleNamespace(model_kwargs={"vlm_max_pixels": 147456}))
        processor = AutoImageProcessor.from_pretrained(ROOT / "reference_image_processor", local_files_only=True)
        errors = []
        for height, width in [(1500, 1500), (576, 1024), (1184, 880), (128, 128)]:
            source = torch.rand(1, 3, height, width)
            expected = processor(images=ref(fake, [source]), do_rescale=False, return_tensors="pt")
            vision, metadata = reference_images.prepare_vision(source)
            actual, grid = process_qwen2vl_images(vision, patch_size=16, image_mean=[0.5] * 3, image_std=[0.5] * 3)
            repeat = processor(images=ref(fake, [source]), do_rescale=False, return_tensors="pt")
            torch.testing.assert_close(expected["pixel_values"], repeat["pixel_values"], rtol=0, atol=0)
            torch.testing.assert_close(grid, expected["image_grid_thw"], rtol=0, atol=0)
            torch.testing.assert_close(actual, expected["pixel_values"], rtol=0, atol=0)
            self.assertEqual(metadata["vision_tokens"], int(grid.prod() // 4))
            errors.append(float((actual - expected["pixel_values"]).abs().max()))
        METRICS["vision_patch_max_abs_error"] = errors

    def test_reference_coordinates_match_training(self):
        ref = reference_function("transformer.py", "_image_coords", {"torch": torch})
        for height, width, text_length in [(2, 3, 7), (64, 64, 150), (74, 55, 281)]:
            torch.testing.assert_close(model_patch.image_coords(1, text_length, height, width, "cpu"), ref(1, text_length, height, width, "cpu"), rtol=0, atol=0)

    def test_time_modulation_matches_training(self):
        reference = reference_function("transformer.py", "_time_modulation", {"torch": torch})
        fake = SimpleNamespace(params=SimpleNamespace(freq_dim=256), time_embedding=torch.nn.Sequential(torch.nn.Linear(256, 128), torch.nn.SiLU(), torch.nn.Linear(128, 128)), shared_block_modulation=torch.nn.Linear(128, 6 * 128))
        sigma = torch.tensor([1.0, 0.731, 0.024])
        expected_mod, expected_time = reference(fake, sigma * 1000, torch.float32)
        actual_time = model_patch.time_vector(fake, sigma, torch.float32)
        torch.testing.assert_close(actual_time, expected_time, rtol=0, atol=0)
        actual_mod = torch.stack(model_patch.modulation(fake, actual_time), dim=1).squeeze(2)
        torch.testing.assert_close(actual_mod, expected_mod, rtol=0, atol=0)

    def test_comfy_flow_euler_matches_velocity_algebra(self):
        from comfy.model_sampling import CONST
        torch.manual_seed(43)
        x, positive, negative = [torch.randn(1, 8, 4, 4) for _ in range(3)]
        flow = CONST()
        errors = []
        tolerance = 16 * torch.finfo(torch.float32).eps * float(x.abs().max())
        for current, following in [(1.0, 0.7), (0.5, 0.1), (0.024, 0.0)]:
            sigma = torch.tensor([current])
            pred = negative + 3 * (positive - negative)
            expected = x + (following - current) * pred
            pos = flow.calculate_denoised(sigma, positive, x)
            neg = flow.calculate_denoised(sigma, negative, x)
            denoised = neg + 3 * (pos - neg)
            actual = x + (following - current) * (x - denoised) / sigma.reshape(-1, 1, 1, 1)
            torch.testing.assert_close(actual, expected, rtol=0, atol=tolerance)
            errors.append(float((actual - expected).abs().max()))
        METRICS["flow_algebra_fp32_max_abs_error"] = errors
        METRICS["flow_algebra_local_atol_from_fp32_epsilon"] = tolerance

    def test_cache_reuse_and_recompute_are_equal(self):
        previous = comfy.model_management.in_training
        comfy.model_management.in_training = True
        try:
            torch.manual_seed(42)
            dit = MOSSImage(dim=128, ffn_dim=192, num_heads=4, num_layers=3, text_dim=32, latent_channels=8, operations=comfy.ops.manual_cast, device="cpu", dtype=torch.float32)
            for parameter in dit.parameters():
                torch.nn.init.normal_(parameter, std=0.07)
            noisy = torch.randn(1, 8, 2, 3)
            context = torch.randn(1, 5, 32)
            references = [torch.randn(1, 8, 2, 2)]
            cached = {"reuse": True, "caches": {}}
            errors = []
            for step in [1.0, 0.5, 0.01]:
                a = model_patch.forward_with_reference(dit, noisy, torch.tensor([step]), context, references, {"cond_or_uncond": [0]}, cached)
                b = model_patch.forward_with_reference(dit, noisy, torch.tensor([step]), context, references, {"cond_or_uncond": [0]}, {"reuse": False, "caches": {}})
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                errors.append(float((a - b).detach().abs().max()))
            self.assertEqual(len(cached["caches"]), 1)
            model_patch.forward_with_reference(dit, noisy, torch.tensor([0.5]), context, references, {"cond_or_uncond": [1]}, cached)
            self.assertEqual(len(cached["caches"]), 2)
            METRICS["cpu_cached_vs_recomputed_max_abs_error"] = errors
        finally:
            comfy.model_management.in_training = previous

    def test_run_cache_released_on_success_and_interruption(self):
        options = {"model_options": {"transformer_options": {}}}
        states = []
        def execute(*values):
            state = values[2]["model_options"]["transformer_options"][model_patch.RUN_KEY]
            states.append(state)
            state["caches"]["temporary"] = torch.ones(3)
            return 7
        self.assertEqual(model_patch.run_lifecycle(execute, None, None, options, None, None), 7)
        self.assertEqual(states[0]["caches"], {})
        self.assertNotIn(model_patch.RUN_KEY, options["model_options"]["transformer_options"])
        def interrupted(*values):
            execute(*values)
            raise RuntimeError("interrupted")
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            model_patch.run_lifecycle(interrupted, None, None, options, None, None)
        self.assertIsNot(states[0], states[1])
        self.assertEqual(states[1]["caches"], {})
        self.assertNotIn(model_patch.RUN_KEY, options["model_options"]["transformer_options"])

    def test_negative_branch_alone_skips_layer_22(self):
        fake = SimpleNamespace()
        fake.set_model_sampler_calc_cond_batch_function = lambda function: setattr(fake, "calculate", function)
        sampling_compat.install_negative_skip(fake)
        original = comfy.samplers.calc_cond_batch
        calls = []
        def calculate(model, conditions, x, sigma, options):
            calls.append((conditions, options))
            return [torch.ones(1, dtype=torch.bfloat16) if value is not None else None for value in conditions]
        comfy.samplers.calc_cond_batch = calculate
        options = {"transformer_options": {}}
        try:
            values = fake.calculate({"model": None, "conds": ["positive", "negative"], "input": None, "sigma": None, "model_options": options})
            self.assertNotIn("patches_replace", calls[0][1]["transformer_options"])
            self.assertEqual(list(calls[1][1]["transformer_options"]["patches_replace"]["dit"]), [("single_block", 22)])
            self.assertTrue(all(value.dtype == torch.float32 for value in values))
            self.assertEqual(options, {"transformer_options": {}})
        finally:
            comfy.samplers.calc_cond_batch = original

    def test_missing_vision_and_unmapped_lora_are_actionable(self):
        with self.assertRaisesRegex(ValueError, "complete Qwen3.5"):
            loaders.audit_vision_keys(["model.language_model.weight"])
        with self.assertRaisesRegex(ValueError, "do not map"):
            loaders.audit_lora_keys({"wrong.lora_A.weight", "wrong.lora_B.weight"}, {})
        self.assertEqual(loaders.audit_lora_keys({"x.lora_A.weight", "x.lora_B.weight"}, {"x": "mapped"})["mapped_modules"], 1)


suite = unittest.defaultTestLoader.loadTestsFromTestCase(ContractChecks)
result = unittest.TextTestRunner(verbosity=2).run(suite)
report = {"tests": result.testsRun, "passed": result.wasSuccessful(), "failures": [str(test) for test, _ in result.failures], "errors": [str(test) for test, _ in result.errors], "metrics": METRICS, "Base_behavior_acceptance": "pending manual acceptance on the training machine"}
Path(args.report).parent.mkdir(parents=True, exist_ok=True)
Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
sys.exit(0 if result.wasSuccessful() else 1)
