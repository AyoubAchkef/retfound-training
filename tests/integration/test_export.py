"""
Export Integration Tests
========================

Test model export functionality.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import json

from retfound.core.config import RETFoundConfig
from retfound.models.factory import create_model
from retfound.export.exporter import ModelExporter
from retfound.export.onnx import ONNXExporter
from retfound.export.torchscript import TorchScriptExporter
from retfound.export.inference import InferenceEngine


@pytest.mark.integration
class TestModelExport:
    """Test model export functionality"""
    
    def test_torchscript_export(self, sample_model, minimal_config, temp_output_dir):
        """Test TorchScript export"""
        # Create exporter
        exporter = TorchScriptExporter()
        
        # Export model
        export_path = temp_output_dir / 'model.pt'
        metadata = {
            'num_classes': minimal_config.num_classes,
            'input_size': minimal_config.input_size,
            'model_type': minimal_config.model_type
        }
        
        exporter.export(
            model=sample_model,
            export_path=export_path,
            input_shape=(1, 3, minimal_config.input_size, minimal_config.input_size),
            metadata=metadata
        )
        
        # Check file created
        assert export_path.exists()
        
        # Load and test
        loaded_model = torch.jit.load(export_path)
        
        # Test inference
        dummy_input = torch.randn(1, 3, minimal_config.input_size, minimal_config.input_size)
        
        sample_model.eval()
        loaded_model.eval()
        
        with torch.no_grad():
            original_output = sample_model(dummy_input)
            loaded_output = loaded_model(dummy_input)
        
        # Outputs should be very close
        assert torch.allclose(original_output, loaded_output, rtol=1e-5)
    
    def test_onnx_export(self, sample_model, minimal_config, temp_output_dir):
        """Test ONNX export"""
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
        
        # Create exporter
        exporter = ONNXExporter(
            opset_version=14,
            simplify=True
        )
        
        # Export model
        export_path = temp_output_dir / 'model.onnx'
        
        exporter.export(
            model=sample_model,
            export_path=export_path,
            input_shape=(1, 3, minimal_config.input_size, minimal_config.input_size),
            metadata={'num_classes': minimal_config.num_classes}
        )
        
        # Check file created
        assert export_path.exists()
        
        # Validate ONNX model
        import onnx
        onnx_model = onnx.load(str(export_path))
        onnx.checker.check_model(onnx_model)
        
        # Test with ONNX Runtime
        import onnxruntime as ort
        
        session = ort.InferenceSession(str(export_path))
        
        # Test inference
        dummy_input = torch.randn(1, 3, minimal_config.input_size, minimal_config.input_size)
        
        # ONNX Runtime inference
        inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
        onnx_output = session.run(None, inputs)[0]
        
        # PyTorch inference
        sample_model.eval()
        with torch.no_grad():
            pytorch_output = sample_model(dummy_input).numpy()
        
        # Outputs should be close
        np.testing.assert_allclose(onnx_output, pytorch_output, rtol=1e-4)
    
    def test_model_exporter_multiple_formats(self, sample_model, minimal_config, temp_output_dir):
        """Test exporting to multiple formats"""
        # Create exporters
        exporters = {
            'torchscript': TorchScriptExporter(),
            'onnx': ONNXExporter() if pytest.importorskip("onnx", reason="ONNX not available") else None
        }
        
        # Remove None exporters
        exporters = {k: v for k, v in exporters.items() if v is not None}
        
        # Create main exporter
        exporter = ModelExporter(
            model=sample_model,
            config=minimal_config,
            exporters=exporters
        )
        
        # Export to all formats
        for format_name in exporters.keys():
            export_path = exporter.export(
                format_name,
                output_dir=temp_output_dir,
                input_shape=(1, 3, minimal_config.input_size, minimal_config.input_size)
            )
            
            assert export_path.exists()
    
    def test_export_with_preprocessing(self, sample_model, minimal_config, temp_output_dir):
        """Test export with preprocessing included"""
        # Create wrapped model with preprocessing
        class ModelWithPreprocessing(torch.nn.Module):
            def __init__(self, model, mean, std):
                super().__init__()
                self.model = model
                self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
                self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
            
            def forward(self, x):
                # Assume input is 0-255
                x = x / 255.0
                x = (x - self.mean) / self.std
                return self.model(x)
        
        # Wrap model
        wrapped_model = ModelWithPreprocessing(
            sample_model,
            mean=minimal_config.pixel_mean,
            std=minimal_config.pixel_std
        )
        
        # Export
        exporter = TorchScriptExporter()
        export_path = temp_output_dir / 'model_with_preprocessing.pt'
        
        exporter.export(
            model=wrapped_model,
            export_path=export_path,
            input_shape=(1, 3, minimal_config.input_size, minimal_config.input_size)
        )
        
        # Test with raw image input
        loaded_model = torch.jit.load(export_path)
        
        # Create raw image (0-255 range)
        raw_image = torch.randint(0, 256, (1, 3, minimal_config.input_size, minimal_config.input_size)).float()
        
        with torch.no_grad():
            output = loaded_model(raw_image)
        
        assert output.shape[1] == minimal_config.num_classes


@pytest.mark.integration
class TestInferenceEngine:
    """Test inference engine functionality"""
    
    def test_inference_engine_pytorch(self, sample_checkpoint, minimal_config, sample_image_files):
        """Test inference engine with PyTorch model"""
        # Create inference engine
        engine = InferenceEngine(
            checkpoint_path=sample_checkpoint,
            device='cpu'
        )
        
        # Single image inference
        result = engine.predict_single(sample_image_files[0])
        
        assert 'predicted_class' in result
        assert 'confidence' in result
        assert 'probabilities' in result
        assert result['predicted_class'] < minimal_config.num_classes
    
    def test_inference_engine_batch(self, sample_checkpoint, sample_image_files):
        """Test batch inference"""
        engine = InferenceEngine(
            checkpoint_path=sample_checkpoint,
            device='cpu'
        )
        
        # Batch inference
        results = engine.predict_batch(sample_image_files)
        
        assert len(results) == len(sample_image_files)
        for result in results:
            assert 'predicted_class' in result
            assert 'confidence' in result
    
    def test_inference_engine_tta(self, sample_checkpoint, sample_image_files):
        """Test inference with test-time augmentation"""
        engine = InferenceEngine(
            checkpoint_path=sample_checkpoint,
            device='cpu',
            use_tta=True,
            tta_transforms=3
        )
        
        # Single image with TTA
        result = engine.predict_single(sample_image_files[0])
        
        # TTA should still return single prediction
        assert 'predicted_class' in result
        assert 'confidence' in result
        
        # Confidence might be different due to averaging
        assert 0 <= result['confidence'] <= 1


@pytest.mark.integration
class TestExportValidation:
    """Test export validation and comparison"""
    
    def test_compare_export_formats(self, sample_model, minimal_config, temp_output_dir):
        """Test comparing outputs across export formats"""
        sample_model.eval()
        
        # Export to multiple formats
        torchscript_path = temp_output_dir / 'model.pt'
        
        # TorchScript export
        ts_exporter = TorchScriptExporter()
        ts_exporter.export(
            sample_model,
            torchscript_path,
            (1, 3, minimal_config.input_size, minimal_config.input_size)
        )
        
        # Load models
        ts_model = torch.jit.load(torchscript_path)
        
        # Test with multiple inputs
        for i in range(5):
            dummy_input = torch.randn(1, 3, minimal_config.input_size, minimal_config.input_size)
            
            with torch.no_grad():
                orig_output = sample_model(dummy_input)
                ts_output = ts_model(dummy_input)
            
            # Check outputs match
            assert torch.allclose(orig_output, ts_output, rtol=1e-5)
    
    def test_export_metadata_preservation(self, sample_checkpoint, temp_output_dir):
        """Test metadata preservation during export"""
        # Load checkpoint
        checkpoint = torch.load(sample_checkpoint, map_location='cpu')
        
        # Create metadata
        metadata = {
            'model_version': '1.0',
            'training_epochs': checkpoint.get('epoch', 0),
            'class_names': checkpoint.get('class_names', []),
            'metrics': checkpoint.get('metrics', {}),
            'config': checkpoint.get('config', {})
        }
        
        # Export with metadata
        export_path = temp_output_dir / 'model_with_metadata.pth'
        
        torch.save({
            'model_state_dict': checkpoint['model_state_dict'],
            'metadata': metadata
        }, export_path)
        
        # Load and verify metadata
        loaded = torch.load(export_path, map_location='cpu')
        
        assert 'metadata' in loaded
        assert loaded['metadata']['model_version'] == '1.0'
        assert 'class_names' in loaded['metadata']


@pytest.mark.integration
class TestProductionExport:
    """Test production-ready export scenarios"""
    
    def test_optimized_export(self, sample_model, minimal_config, temp_output_dir):
        """Test optimized model export"""
        # Optimize model for inference
        sample_model.eval()
        
        # Fuse modules if possible
        if hasattr(torch.nn.utils, 'fusion'):
            sample_model = torch.nn.utils.fusion.fuse_conv_bn_eval(sample_model)
        
        # Export optimized model
        exporter = TorchScriptExporter(optimize=True)
        export_path = temp_output_dir / 'model_optimized.pt'
        
        exporter.export(
            model=sample_model,
            export_path=export_path,
            input_shape=(1, 3, minimal_config.input_size, minimal_config.input_size)
        )
        
        # Check optimization
        loaded_model = torch.jit.load(export_path)
        
        # Optimized model should work correctly
        dummy_input = torch.randn(1, 3, minimal_config.input_size, minimal_config.input_size)
        with torch.no_grad():
            output = loaded_model(dummy_input)
        
        assert output.shape[1] == minimal_config.num_classes
    
    def test_quantized_export(self, sample_model, minimal_config, temp_output_dir):
        """Test quantized model export"""
        pytest.importorskip("torch.quantization")
        
        sample_model.eval()
        
        # Prepare for quantization
        sample_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Prepare model
        prepared_model = torch.quantization.prepare(sample_model, inplace=False)
        
        # Calibrate with dummy data
        for _ in range(10):
            dummy_input = torch.randn(1, 3, minimal_config.input_size, minimal_config.input_size)
            prepared_model(dummy_input)
        
        # Convert to quantized
        quantized_model = torch.quantization.convert(prepared_model, inplace=False)
        
        # Export quantized model
        export_path = temp_output_dir / 'model_quantized.pth'
        torch.save({
            'model': quantized_model,
            'config': minimal_config.to_dict()
        }, export_path)
        
        # Check size reduction
        original_size = sum(p.numel() * p.element_size() for p in sample_model.parameters())
        quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())
        
        # Quantized should be smaller
        assert quantized_size < original_size
    
    def test_mobile_export(self, sample_model, minimal_config, temp_output_dir):
        """Test mobile-optimized export"""
        sample_model.eval()
        
        # Export for mobile
        exporter = TorchScriptExporter(optimize_for_mobile=True)
        export_path = temp_output_dir / 'model_mobile.ptl'
        
        # Create mobile-friendly input shape
        mobile_input_shape = (1, 3, 224, 224)  # Standard mobile size
        
        exporter.export(
            model=sample_model,
            export_path=export_path,
            input_shape=mobile_input_shape,
            metadata={
                'platform': 'mobile',
                'optimized': True
            }
        )
        
        # Mobile model should be created
        assert export_path.exists()
