"""
Unit tests for Attention U-Net Architecture and Segmentation Inference.
"""
import unittest
import torch
import numpy as np

from vision_ai.models.attention_unet import AttentionUNet, AttentionGate
from vision_ai.models.model_factory import OilSpillModelInference

class TestAttentionUNet(unittest.TestCase):

    def test_attention_gate_shape(self):
        # F_g=64, F_l=64, F_int=32
        att_gate = AttentionGate(F_g=64, F_l=64, F_int=32)
        g = torch.randn(2, 64, 16, 16)
        x = torch.randn(2, 64, 16, 16)
        out = att_gate(g, x)
        self.assertEqual(out.shape, (2, 64, 16, 16))

    def test_attention_unet_forward_pass(self):
        # 2-channel input (VV, VH)
        model = AttentionUNet(in_channels=2, out_channels=1, base_filters=16)
        dummy_input = torch.randn(2, 2, 128, 128)
        output = model(dummy_input)
        
        self.assertEqual(output.shape, (2, 1, 128, 128))

    def test_model_inference_scene(self):
        inference_runner = OilSpillModelInference(
            in_channels=2,
            base_filters=16,
            tile_size=128,
            overlap=32,
            device="cpu"
        )
        dummy_scene = np.random.uniform(0.0, 1.0, size=(2, 200, 200)).astype(np.float32)
        prob_mask, binary_mask = inference_runner.predict_scene(dummy_scene, threshold=0.5)
        
        self.assertEqual(prob_mask.shape, (200, 200))
        self.assertEqual(binary_mask.shape, (200, 200))
        self.assertTrue(np.all((prob_mask >= 0.0) & (prob_mask <= 1.0)))
        self.assertTrue(set(np.unique(binary_mask)).issubset({0, 1}))

if __name__ == "__main__":
    unittest.main()
