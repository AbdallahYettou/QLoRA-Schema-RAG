import json
from pathlib import Path
import struct
import tempfile
import unittest

from scripts.publish_adapter import inspect_adapter


class AdapterValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "adapter_config.json").write_text(json.dumps({
            "peft_type": "LORA", "r": 1, "lora_alpha": 1,
            "base_model_name_or_path": "example/base", "target_modules": ["q_proj"]}))

    def tearDown(self):
        self.temp.cleanup()

    def write_weights(self, name="model.q_proj.lora_A.weight", payload=b"\0" * 8):
        header = json.dumps({name: {"dtype": "F32", "shape": [1, 2], "data_offsets": [0, 8]}}).encode()
        (self.root / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + payload)

    def test_valid_adapter(self):
        self.write_weights()
        _, info = inspect_adapter(self.root)
        self.assertEqual(info["parameters"], 2)
        self.assertEqual(info["tensor_dtypes"], ["F32"])

    def test_reject_full_model(self):
        self.write_weights(name="model.q_proj.weight")
        with self.assertRaisesRegex(ValueError, "adapter-only"):
            inspect_adapter(self.root)

    def test_reject_partial_download(self):
        self.write_weights(payload=b"\0")
        with self.assertRaisesRegex(ValueError, "truncated"):
            inspect_adapter(self.root)


if __name__ == "__main__":
    unittest.main()
