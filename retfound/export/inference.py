"""
Inference Utilities
==================

Utilities for model inference and deployment.
"""

import logging
from pathlib import Path
from typing import Union, Optional, Dict, Any, List, Tuple
import time
import json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False


class InferenceModel:
    """
    Unified interface for inference across different model formats
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = 'cuda',
        input_shape: Optional[Tuple[int, ...]] = None,
        normalize: bool = True,
        mean: List[float] = [0.5, 0.5, 0.5],
        std: List[float] = [0.5, 0.5, 0.5]
    ):
        """
        Initialize inference model
        
        Args:
            model_path: Path to model file
            device: Device to run on
            input_shape: Expected input shape (C, H, W)
            normalize: Whether to normalize inputs
            mean: Normalization mean
            std: Normalization std
        """
        self.model_path = Path(model_path)
        self.device = device
        self.input_shape = input_shape or (3, 224, 224)
        self.normalize = normalize
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        
        # Load model based on format
        self._load_model()
        
        # Warm up
        self._warmup()
    
    def _load_model(self):
        """Load model based on file format"""
        suffix = self.model_path.suffix.lower()
        
        if suffix in ['.pth', '.pt']:
            self._load_pytorch()
        elif suffix == '.onnx' and ONNX_AVAILABLE:
            self._load_onnx()
        elif suffix == '.engine' and TENSORRT_AVAILABLE:
            self._load_tensorrt()
        else:
            raise ValueError(f"Unsupported model format: {suffix}")
    
    def _load_pytorch(self):
        """Load PyTorch or TorchScript model"""
        logger.info(f"Loading PyTorch model from {self.model_path}")
        
        if self.model_path.name.endswith(('_traced.pt', '_scripted.pt')):
            # TorchScript model
            self.model = torch.jit.load(str(self.model_path))
            self.model_type = 'torchscript'
        else:
            # Regular PyTorch model
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Extract model state dict
            if 'model_state_dict' in checkpoint:
                raise ValueError(
                    "PyTorch checkpoint found but model class not provided. "
                    "Use TorchScript format for standalone inference."
                )
            else:
                # Assume it's a TorchScript model
                self.model = torch.jit.load(str(self.model_path))
                self.model_type = 'torchscript'
        
        self.model.eval()
        self.model.to(self.device)
    
    def _load_onnx(self):
        """Load ONNX model"""
        logger.info(f"Loading ONNX model from {self.model_path}")
        
        # Create ONNX runtime session
        providers = ['CUDAExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        self.model_type = 'onnx'
        
        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
    
    def _load_tensorrt(self):
        """Load TensorRT engine"""
        logger.info(f"Loading TensorRT engine from {self.model_path}")
        
        # Load engine
        with open(self.model_path, 'rb') as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.model_type = 'tensorrt'
        
        # Setup buffers
        self._setup_trt_buffers()
    
    def _setup_trt_buffers(self):
        """Setup TensorRT buffers"""
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = torch.cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            host_mem = np.empty(size, dtype=dtype)
            device_mem = torch.cuda.ByteTensor(host_mem.nbytes)
            
            self.bindings.append(int(device_mem.data_ptr()))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
    
    def _warmup(self, num_runs: int = 3):
        """Warm up model"""
        logger.info("Warming up model...")
        
        dummy_input = torch.randn(1, *self.input_shape)
        
        for _ in range(num_runs):
            _ = self.predict_tensor(dummy_input)
    
    def preprocess(self, image: Union[str, Path, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Preprocess image for inference
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            
        Returns:
            Preprocessed tensor
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        
        # Convert to PIL if numpy
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:  # Grayscale
                image = np.stack([image] * 3, axis=-1)
            image = Image.fromarray(image.astype(np.uint8))
        
        # Resize
        size = (self.input_shape[1], self.input_shape[2])
        image = image.resize(size, Image.Resampling.LANCZOS)
        
        # Convert to tensor
        image = torch.from_numpy(np.array(image)).float()
        image = image.permute(2, 0, 1)  # HWC -> CHW
        
        # Normalize to [0, 1]
        image = image / 255.0
        
        # Apply normalization if requested
        if self.normalize:
            image = (image - self.mean) / self.std
        
        # Add batch dimension
        return image.unsqueeze(0)
    
    def predict_tensor(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Run inference on tensor
        
        Args:
            input_tensor: Input tensor (B, C, H, W)
            
        Returns:
            Output tensor
        """
        if self.model_type in ['pytorch', 'torchscript']:
            return self._predict_pytorch(input_tensor)
        elif self.model_type == 'onnx':
            return self._predict_onnx(input_tensor)
        elif self.model_type == 'tensorrt':
            return self._predict_tensorrt(input_tensor)
    
    def _predict_pytorch(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """PyTorch/TorchScript inference"""
        input_tensor = input_tensor.to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
        
        return output
    
    def _predict_onnx(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """ONNX inference"""
        # Convert to numpy
        input_numpy = input_tensor.cpu().numpy()
        
        # Run inference
        outputs = self.session.run(
            self.output_names,
            {self.input_name: input_numpy}
        )
        
        # Convert back to tensor
        return torch.from_numpy(outputs[0])
    
    def _predict_tensorrt(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """TensorRT inference"""
        # Prepare input
        input_numpy = input_tensor.cpu().numpy().ravel()
        self.inputs[0]['host'] = input_numpy
        
        # Transfer to device
        with self.stream:
            self.inputs[0]['device'].copy_(
                torch.from_numpy(self.inputs[0]['host']).cuda()
            )
            
            # Run inference
            self.context.execute_async_v2(
                bindings=self.bindings,
                stream_handle=self.stream.cuda_stream
            )
            
            # Transfer outputs
            for out in self.outputs:
                torch.cuda.memcpy_async(
                    out['host'], out['device'],
                    out['host'].nbytes,
                    torch.cuda.memcpyDeviceToHost,
                    self.stream
                )
            
            self.stream.synchronize()
        
        # Reshape and convert to tensor
        output_shape = input_tensor.shape[0], -1  # Batch size, features
        output = self.outputs[0]['host'].reshape(output_shape)
        
        return torch.from_numpy(output)
    
    def predict(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        return_probs: bool = True
    ) -> Dict[str, Any]:
        """
        Run inference on image
        
        Args:
            image: Input image
            return_probs: Whether to return probabilities
            
        Returns:
            Dictionary with predictions
        """
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Predict
        start_time = time.time()
        output = self.predict_tensor(input_tensor)
        inference_time = time.time() - start_time
        
        # Process output
        if return_probs:
            probs = torch.softmax(output, dim=1)
            pred_class = probs.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()
            
            result = {
                'predicted_class': pred_class,
                'confidence': confidence,
                'probabilities': probs[0].cpu().numpy().tolist(),
                'inference_time_ms': inference_time * 1000
            }
        else:
            result = {
                'logits': output[0].cpu().numpy().tolist(),
                'inference_time_ms': inference_time * 1000
            }
        
        return result
    
    def benchmark(
        self,
        input_shape: Optional[Tuple[int, ...]] = None,
        batch_sizes: List[int] = [1, 4, 8, 16, 32],
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[int, Dict[str, float]]:
        """
        Benchmark model performance
        
        Args:
            input_shape: Input shape (defaults to model's shape)
            batch_sizes: Batch sizes to test
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs
            
        Returns:
            Benchmark results per batch size
        """
        if input_shape is None:
            input_shape = self.input_shape
        
        results = {}
        
        for batch_size in batch_sizes:
            logger.info(f"Benchmarking batch size {batch_size}...")
            
            # Create dummy input
            dummy_input = torch.randn(batch_size, *input_shape)
            
            # Warmup
            for _ in range(warmup_runs):
                _ = self.predict_tensor(dummy_input)
            
            # Benchmark
            times = []
            for _ in range(num_runs):
                start_time = time.perf_counter()
                _ = self.predict_tensor(dummy_input)
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # ms
            
            times = np.array(times)
            
            results[batch_size] = {
                'mean_ms': np.mean(times),
                'std_ms': np.std(times),
                'min_ms': np.min(times),
                'max_ms': np.max(times),
                'p95_ms': np.percentile(times, 95),
                'p99_ms': np.percentile(times, 99),
                'throughput_fps': (batch_size * 1000) / np.mean(times)
            }
        
        return results


def create_inference_script(
    model_name: str,
    input_shape: Tuple[int, ...],
    available_formats: List[Any],
    metadata: Dict[str, Any]
) -> str:
    """
    Create standalone inference script
    
    Args:
        model_name: Name of the model
        input_shape: Input shape (C, H, W)
        available_formats: Available model formats
        metadata: Model metadata
        
    Returns:
        Inference script content
    """
    # Convert format enums to strings
    format_strings = [f.value if hasattr(f, 'value') else str(f) for f in available_formats]
    
    script = f'''#!/usr/bin/env python3
"""
Inference script for {model_name}
Auto-generated by RETFound Export System
"""

import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from PIL import Image

# Model configuration
MODEL_NAME = "{model_name}"
INPUT_SHAPE = {input_shape}
NORMALIZE_MEAN = {metadata.get('normalize_mean', [0.5, 0.5, 0.5])}
NORMALIZE_STD = {metadata.get('normalize_std', [0.5, 0.5, 0.5])}
CLASS_NAMES = {metadata.get('class_names', [])}


class InferenceModel:
    """Simple inference wrapper"""
    
    def __init__(self, model_path, device='cuda'):
        self.model_path = Path(model_path)
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Load model
        if self.model_path.suffix in ['.pt', '.pth']:
            self.model = torch.jit.load(str(model_path))
            self.model.eval()
            self.model.to(self.device)
        else:
            raise ValueError(f"Unsupported format: {{self.model_path.suffix}}")
        
        # Normalization
        self.mean = torch.tensor(NORMALIZE_MEAN).view(3, 1, 1)
        self.std = torch.tensor(NORMALIZE_STD).view(3, 1, 1)
    
    def preprocess(self, image_path):
        """Preprocess image"""
        # Load and resize
        image = Image.open(image_path).convert('RGB')
        image = image.resize((INPUT_SHAPE[1], INPUT_SHAPE[2]), Image.Resampling.LANCZOS)
        
        # Convert to tensor
        image = torch.from_numpy(np.array(image)).float()
        image = image.permute(2, 0, 1) / 255.0
        
        # Normalize
        image = (image - self.mean) / self.std
        
        return image.unsqueeze(0)
    
    def predict(self, image_path):
        """Run prediction"""
        # Preprocess
        input_tensor = self.preprocess(image_path).to(self.device)
        
        # Predict
        start_time = time.time()
        with torch.no_grad():
            output = self.model(input_tensor)
        inference_time = time.time() - start_time
        
        # Process output
        probs = torch.softmax(output, dim=1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
        
        result = {{
            'image': str(image_path),
            'predicted_class': pred_class,
            'predicted_label': CLASS_NAMES[pred_class] if CLASS_NAMES else f"Class {{pred_class}}",
            'confidence': confidence,
            'inference_time_ms': inference_time * 1000
        }}
        
        return result


def main():
    parser = argparse.ArgumentParser(description=f'Inference for {{MODEL_NAME}}')
    parser.add_argument('input', help='Input image or directory')
    parser.add_argument('--model', default=f'{{MODEL_NAME}}_traced.pt', help='Model file')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--output', help='Output JSON file')
    parser.add_argument('--batch', action='store_true', help='Process directory')
    
    args = parser.parse_args()
    
    # Initialize model
    model = InferenceModel(args.model, args.device)
    
    # Process input
    input_path = Path(args.input)
    
    if args.batch and input_path.is_dir():
        # Process directory
        results = []
        image_files = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))
        
        print(f"Processing {{len(image_files)}} images...")
        for image_file in image_files:
            result = model.predict(image_file)
            results.append(result)
            print(f"{{image_file.name}}: {{result['predicted_label']}} ({{result['confidence']:.2%}})")
        
        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
    
    else:
        # Single image
        result = model.predict(input_path)
        print(f"\\nPrediction: {{result['predicted_label']}}")
        print(f"Confidence: {{result['confidence']:.2%}}")
        print(f"Inference time: {{result['inference_time_ms']:.1f}} ms")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()
'''
    
    return script


def benchmark_inference(
    model_path: Union[str, Path],
    input_shape: Tuple[int, ...] = (3, 224, 224),
    batch_sizes: List[int] = [1, 4, 8, 16, 32],
    device: str = 'cuda',
    num_runs: int = 100
) -> Dict[str, Any]:
    """
    Benchmark inference performance
    
    Args:
        model_path: Path to model
        input_shape: Input shape
        batch_sizes: Batch sizes to test
        device: Device to run on
        num_runs: Number of runs per batch size
        
    Returns:
        Benchmark results
    """
    logger.info(f"Benchmarking {model_path}")
    
    # Create inference model
    model = InferenceModel(model_path, device=device, input_shape=input_shape)
    
    # Run benchmark
    results = model.benchmark(
        input_shape=input_shape,
        batch_sizes=batch_sizes,
        num_runs=num_runs
    )
    
    # Summary
    logger.info("\nBenchmark Results:")
    logger.info(f"{'Batch':>6} {'Mean (ms)':>10} {'Std (ms)':>10} {'FPS':>10}")
    logger.info("-" * 40)
    
    for batch_size, stats in results.items():
        logger.info(
            f"{batch_size:>6} {stats['mean_ms']:>10.2f} "
            f"{stats['std_ms']:>10.2f} {stats['throughput_fps']:>10.1f}"
        )
    
    return {
        'model_path': str(model_path),
        'model_type': model.model_type,
        'device': device,
        'input_shape': input_shape,
        'results': results
    }