"""CPU-only regression checks for effective batch size accounting."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "training_config", Path(__file__).parents[1] / "rldx/configs/training/training_config.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
TrainingConfig = module.TrainingConfig


class EffectiveBatchTest(unittest.TestCase):
    def test_accumulation_preserves_global_batch(self):
        for accumulation in (1, 2, 4, 8, 16, 32):
            cfg = TrainingConfig(global_batch_size=64, num_gpus=2,
                                 gradient_accumulation_steps=accumulation)
            self.assertEqual(cfg.per_device_batch_size() * 2 * accumulation, 64)

    def test_invalid_batches_fail(self):
        for kwargs in ({"global_batch_size": 63}, {"global_batch_size": 0},
                       {"gradient_accumulation_steps": 3},
                       {"gradient_accumulation_steps": 0}, {"num_gpus": 0}):
            cfg = TrainingConfig(global_batch_size=64, num_gpus=2)
            for key, value in kwargs.items():
                setattr(cfg, key, value)
            with self.assertRaises(ValueError):
                cfg.per_device_batch_size()

    def test_legacy_microbatch_override(self):
        cfg = TrainingConfig(batch_size=3, num_gpus=2, gradient_accumulation_steps=8)
        self.assertEqual(cfg.per_device_batch_size(), 3)


if __name__ == "__main__":
    unittest.main()
