"""Regression checks for native H3 graphs without loading GPU weights."""
import pathlib
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "webui"))
import server as S


class NativeGraphTests(unittest.TestCase):
    def test_i2v_uses_selected_full_model_and_first_frame(self):
        with patch.object(S, 'NATIVE_PIPELINE', True), patch.object(S, 'TEXT_ENCODER', 'encoder_bf16.safetensors'):
            graph = S.build_graph('test', {'model': 'full', 'low_vram': True, 'sage': True}, 'first.png', 864, 480, 124)
        classes = {n['class_type'] for n in graph.values()}
        self.assertIn('MiniMaxH3ImageToVideo', classes)
        self.assertNotIn('MiniMaxH3ReferenceToVideo', classes)
        self.assertFalse(any('T8' in c or c.startswith('VHS_') or 'LowVRAM' in c or 'Sage' in c for c in classes))
        self.assertEqual(graph['4']['inputs']['unet_name'], S.MODELS['full'])
        self.assertEqual(graph['3']['inputs']['clip_name'], 'encoder_bf16.safetensors')
        ref_id, _ = graph['6']['inputs']['first_frame']
        self.assertEqual(graph[ref_id]['inputs']['image'], 'first.png')
        self.assertEqual(graph['9']['inputs']['steps'], 20)
        self.assertEqual(graph['15']['inputs']['format.codec'], 'h264')

    def test_ref2va_keeps_references_and_custom_lora_sampling(self):
        params = {'steps': 23, 'seed': 0, 'turbo_lora': True, 'lora_name': 'adapter.safetensors',
                  'lora_strength': 0.5, 'sampler': 'euler', 'scheduler': 'beta'}
        with patch.object(S, 'NATIVE_PIPELINE', True), patch.object(S, 'find_ref2va_model', return_value='reference_bf16.safetensors'):
            graph = S.build_ref2va_graph('test', params, ['one.png', 'two.png'], 864, 480, 124, ['voice.wav'])
        self.assertEqual(graph['4']['inputs']['unet_name'], 'reference_bf16.safetensors')
        self.assertEqual(graph['5']['inputs']['lora_name'], 'adapter.safetensors')
        self.assertEqual(graph['5']['inputs']['strength_model'], 0.5)
        self.assertEqual(graph['5s']['inputs']['model'], ['5', 0])
        self.assertEqual(graph['8']['inputs']['sampler_name'], 'euler')
        self.assertEqual(graph['9']['inputs']['scheduler'], 'beta')
        self.assertEqual(graph['9']['inputs']['steps'], 23)
        self.assertEqual(graph['7']['inputs']['noise_seed'], 0)
        self.assertEqual(graph['6']['inputs']['ref_images.ref_image_1'], ['21', 0])
        self.assertEqual(graph['6']['inputs']['ref_audios.ref_audio_0'], ['30', 0])

    def test_legacy_i2v_remains_available(self):
        with patch.object(S, 'NATIVE_PIPELINE', False):
            graph = S.build_graph('test', {}, 'first.png', 864, 480, 124)
        classes = {n['class_type'] for n in graph.values()}
        self.assertIn('MiniMaxH3AudioConditioningT8', classes)
        self.assertIn('VHS_VideoCombine', classes)


if __name__ == '__main__':
    unittest.main()
