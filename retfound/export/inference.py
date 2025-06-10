"""
Inference utilities for exported RETFound models.
Supports inference on v6.1 models with 28 classes.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2

from ..core.constants import DATASET_V61_CLASSES, DATASET_V40_CLASSES, CRITICAL_CONDITIONS
from ..core.exceptions import InferenceError
from ..data.transforms import get_eval_transforms
from ..utils.logging import get_logger

logger = get_logger(__name__)


class RETFoundPredictor:
    """Inference engine for exported RETFound models."""
    
    def __init__(
        self,
        model_path: str,
        metadata_path: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 1
    ):
        """
        Initialize predictor with exported model.
        
        Args:
            model_path: Path to exported model (.onnx, .pt, .trt)
            metadata_path: Optional path to metadata JSON
            device: Device for inference ('cpu', 'cuda', 'tensorrt')
            batch_size: Batch size for inference
        """
        self.model_path = Path(model_path)
        self.batch_size = batch_size
        
        # Auto-detect format
        self.format = self._detect_format()
        
        # Load metadata
        self.metadata = self._load_metadata(metadata_path)
        
        # Extract model info
        self.dataset_version = self.metadata.get('dataset_version', 'v6.1')
        self.num_classes = self.metadata.get('num_classes', 28)
        self.class_names = self.metadata.get('class_names', 
                                             DATASET_V61_CLASSES if self.num_classes == 28 else DATASET_V40_CLASSES)
        
        # Set device
        self.device = self._setup_device(device)
        
        # Load model
        self.model = self._load_model()
        
        # Setup preprocessing
        self.transform = self._setup_preprocessing()
        
        logger.info(f"Initialized {self.format} predictor for {self.dataset_version} "
                   f"with {self.num_classes} classes on {self.device}")
    
    def _detect_format(self) -> str:
        """Detect model format from file extension."""
        suffix = self.model_path.suffix.lower()
        if suffix == '.onnx':
            return 'onnx'
        elif suffix in ['.pt', '.pth']:
            return 'torchscript'
        elif suffix == '.trt':
            return 'tensorrt'
        else:
            raise InferenceError(f"Unsupported model format: {suffix}")
    
    def _load_metadata(self, metadata_path: Optional[str]) -> Dict:
        """Load model metadata."""
        # Try explicit metadata path
        if metadata_path:
            with open(metadata_path) as f:
                return json.load(f).get('model_info', {})
        
        # Try default metadata locations
        default_paths = [
            self.model_path.parent / "export_metadata.json",
            self.model_path.with_suffix('.json')
        ]
        
        for path in default_paths:
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                    return data.get('model_info', data)
        
        # Try to extract from model if ONNX
        if self.format == 'onnx':
            metadata = self._extract_onnx_metadata()
            if metadata:
                return metadata
        
        logger.warning("No metadata found, using defaults")
        return {}
    
    def _extract_onnx_metadata(self) -> Dict:
        """Extract metadata from ONNX model."""
        try:
            import onnx
            
            model = onnx.load(str(self.model_path))
            metadata = {}
            
            for prop in model.metadata_props:
                if prop.key == 'class_names':
                    metadata['class_names'] = json.loads(prop.value)
                elif prop.key == 'num_classes':
                    metadata['num_classes'] = int(prop.value)
                elif prop.key == 'dataset_version':
                    metadata['dataset_version'] = prop.value
            
            return metadata
        except:
            return {}
    
    def _setup_device(self, device: Optional[str]) -> str:
        """Setup inference device."""
        if device:
            return device
        
        # Auto-detect based on format and availability
        if self.format == 'tensorrt':
            return 'tensorrt'
        elif torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
    
    def _load_model(self):
        """Load model based on format."""
        if self.format == 'onnx':
            return self._load_onnx_model()
        elif self.format == 'torchscript':
            return self._load_torchscript_model()
        elif self.format == 'tensorrt':
            return self._load_tensorrt_model()
    
    def _load_onnx_model(self):
        """Load ONNX model."""
        try:
            import onnxruntime as ort
        except ImportError:
            raise InferenceError("ONNX inference requires onnxruntime")
        
        # Set providers based on device
        if self.device == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        # Create inference session
        session = ort.InferenceSession(str(self.model_path), providers=providers)
        
        return session
    
    def _load_torchscript_model(self):
        """Load TorchScript model."""
        model = torch.jit.load(str(self.model_path))
        model.eval()
        
        # Move to device
        if self.device == 'cuda':
            model = model.cuda()
        
        # Extract metadata if available
        if hasattr(model, 'dataset_version'):
            self.dataset_version = model.dataset_version
        if hasattr(model, 'num_classes'):
            self.num_classes = model.num_classes
        if hasattr(model, 'class_names'):
            self.class_names = model.class_names
        
        return model
    
    def _load_tensorrt_model(self):
        """Load TensorRT model."""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
        except ImportError:
            raise InferenceError("TensorRT inference requires tensorrt and pycuda")
        
        # Load engine
        with open(self.model_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(engine_data)
        context = engine.create_execution_context()
        
        return {'engine': engine, 'context': context}
    
    def _setup_preprocessing(self):
        """Setup image preprocessing pipeline."""
        # Get input size from metadata or use default
        input_size = self.metadata.get('input_size', 224)
        
        # Use standard eval transforms
        return get_eval_transforms(input_size=input_size)
    
    def predict(
        self,
        images: Union[str, Path, Image.Image, np.ndarray, List],
        return_probabilities: bool = True,
        return_features: bool = False,
        check_critical: bool = True
    ) -> Union[Dict, List[Dict]]:
        """
        Run inference on input images.
        
        Args:
            images: Input image(s) - path, PIL Image, numpy array, or list
            return_probabilities: Whether to return class probabilities
            return_features: Whether to return feature embeddings
            check_critical: Whether to check for critical conditions
            
        Returns:
            Prediction dictionary or list of dictionaries
        """
        # Handle single image
        if not isinstance(images, list):
            images = [images]
            single_image = True
        else:
            single_image = False
        
        # Preprocess images
        processed_images = []
        image_metadata = []
        
        for img in images:
            processed, metadata = self._preprocess_image(img)
            processed_images.append(processed)
            image_metadata.append(metadata)
        
        # Batch inference
        results = []
        for i in range(0, len(processed_images), self.batch_size):
            batch = processed_images[i:i + self.batch_size]
            batch_metadata = image_metadata[i:i + self.batch_size]
            
            # Stack into batch tensor
            if self.format == 'onnx':
                batch_tensor = np.stack(batch)
            else:
                batch_tensor = torch.stack([torch.from_numpy(img) for img in batch])
                if self.device == 'cuda':
                    batch_tensor = batch_tensor.cuda()
            
            # Run inference
            outputs = self._run_inference(batch_tensor, return_features)
            
            # Process outputs
            batch_results = self._process_outputs(
                outputs, 
                batch_metadata,
                return_probabilities,
                return_features,
                check_critical
            )
            
            results.extend(batch_results)
        
        # Return single result if single image
        return results[0] if single_image else results
    
    def _preprocess_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> Tuple[np.ndarray, Dict]:
        """Preprocess single image."""
        metadata = {}
        
        # Load image if path
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            metadata['path'] = str(image_path)
            metadata['filename'] = image_path.name
            image = Image.open(image_path).convert('RGB')
        elif isinstance(image, np.ndarray):
            # Convert numpy to PIL
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise InferenceError(f"Unsupported image type: {type(image)}")
        
        # Store original size
        metadata['original_size'] = image.size
        
        # Apply transforms
        transformed = self.transform(image)
        
        # Convert to numpy
        if isinstance(transformed, torch.Tensor):
            transformed = transformed.numpy()
        
        return transformed, metadata
    
    def _run_inference(self, batch: Union[np.ndarray, torch.Tensor], return_features: bool) -> Dict:
        """Run model inference on batch."""
        if self.format == 'onnx':
            return self._run_onnx_inference(batch)
        elif self.format == 'torchscript':
            return self._run_torchscript_inference(batch, return_features)
        elif self.format == 'tensorrt':
            return self._run_tensorrt_inference(batch)
    
    def _run_onnx_inference(self, batch: np.ndarray) -> Dict:
        """Run ONNX inference."""
        input_name = self.model.get_inputs()[0].name
        outputs = self.model.run(None, {input_name: batch})
        
        return {'logits': outputs[0]}
    
    def _run_torchscript_inference(self, batch: torch.Tensor, return_features: bool) -> Dict:
        """Run TorchScript inference."""
        with torch.no_grad():
            outputs = self.model(batch)
        
        # Handle different output formats
        if isinstance(outputs, dict):
            return outputs
        else:
            # Assume outputs are logits
            result = {'logits': outputs}
            
            # Try to get features if requested
            if return_features and hasattr(self.model, 'forward_features'):
                features = self.model.forward_features(batch)
                result['features'] = features
            
            return result
    
    def _run_tensorrt_inference(self, batch: Union[np.ndarray, torch.Tensor]) -> Dict:
        """Run TensorRT inference."""
        # Convert to numpy if needed
        if isinstance(batch, torch.Tensor):
            batch = batch.cpu().numpy()
        
        # Implementation depends on TensorRT Python API
        # This is a placeholder
        raise NotImplementedError("TensorRT inference not fully implemented")
    
    def _process_outputs(
        self,
        outputs: Dict,
        metadata: List[Dict],
        return_probabilities: bool,
        return_features: bool,
        check_critical: bool
    ) -> List[Dict]:
        """Process model outputs into results."""
        # Get logits
        logits = outputs['logits']
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)
        
        # Compute probabilities
        probabilities = F.softmax(logits, dim=-1)
        
        # Get predictions
        predictions = logits.argmax(dim=-1)
        
        # Process each sample
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            result = {
                'predicted_class_idx': int(pred),
                'predicted_class': self.class_names[int(pred)],
                'confidence': float(probs[pred]),
                'metadata': metadata[i]
            }
            
            # Add probabilities if requested
            if return_probabilities:
                result['probabilities'] = {
                    self.class_names[j]: float(probs[j])
                    for j in range(self.num_classes)
                }
                
                # Add top-5 predictions
                top5_probs, top5_indices = torch.topk(probs, min(5, self.num_classes))
                result['top5_predictions'] = [
                    {
                        'class': self.class_names[int(idx)],
                        'probability': float(prob)
                    }
                    for prob, idx in zip(top5_probs, top5_indices)
                ]
            
            # Add features if available and requested
            if return_features and 'features' in outputs:
                features = outputs['features'][i]
                if isinstance(features, torch.Tensor):
                    features = features.cpu().numpy()
                result['features'] = features
            
            # Check for critical conditions
            if check_critical and self.dataset_version == "v6.1":
                result['critical_analysis'] = self._check_critical_conditions(
                    int(pred), probs
                )
            
            # Add modality info for v6.1
            if self.dataset_version == "v6.1":
                if int(pred) < 18:
                    result['modality'] = 'fundus'
                else:
                    result['modality'] = 'oct'
            
            results.append(result)
        
        return results
    
    def _check_critical_conditions(self, predicted_class: int, probabilities: torch.Tensor) -> Dict:
        """Check for critical conditions in prediction."""
        analysis = {
            'is_critical': False,
            'critical_conditions_detected': []
        }
        
        # Check if predicted class is critical
        predicted_class_name = self.class_names[predicted_class]
        if predicted_class_name in CRITICAL_CONDITIONS:
            analysis['is_critical'] = True
            condition_info = CRITICAL_CONDITIONS[predicted_class_name].copy()
            condition_info['probability'] = float(probabilities[predicted_class])
            analysis['critical_conditions_detected'].append({
                'condition': predicted_class_name,
                **condition_info
            })
        
        # Check for other critical conditions with high probability
        for condition_name, info in CRITICAL_CONDITIONS.items():
            if condition_name in self.class_names and condition_name != predicted_class_name:
                class_idx = self.class_names.index(condition_name)
                prob = float(probabilities[class_idx])
                
                # Flag if probability is above threshold (e.g., 0.1)
                if prob > 0.1:
                    analysis['critical_conditions_detected'].append({
                        'condition': condition_name,
                        'probability': prob,
                        **info
                    })
        
        # Sort by severity and probability
        if analysis['critical_conditions_detected']:
            analysis['critical_conditions_detected'].sort(
                key=lambda x: (x['severity'], x['probability']),
                reverse=True
            )
            analysis['recommendation'] = self._get_recommendation(
                analysis['critical_conditions_detected'][0]
            )
        
        return analysis
    
    def _get_recommendation(self, condition_info: Dict) -> str:
        """Get recommendation based on condition severity and urgency."""
        severity = condition_info['severity']
        urgency = condition_info['urgency']
        
        if severity == 'critical' or urgency == 'immediate':
            return "URGENT: Immediate ophthalmological evaluation required"
        elif severity == 'high' or urgency == 'urgent':
            return "Schedule urgent ophthalmological consultation within 24-48 hours"
        elif severity == 'moderate':
            return "Schedule ophthalmological evaluation within 1-2 weeks"
        else:
            return "Monitor condition and schedule routine ophthalmological follow-up"
    
    def predict_batch(
        self,
        image_paths: List[Union[str, Path]],
        save_results: bool = True,
        output_path: Optional[str] = None
    ) -> List[Dict]:
        """
        Run batch prediction on multiple images.
        
        Args:
            image_paths: List of image paths
            save_results: Whether to save results to file
            output_path: Path to save results
            
        Returns:
            List of prediction dictionaries
        """
        logger.info(f"Running batch prediction on {len(image_paths)} images...")
        
        # Run predictions
        results = self.predict(image_paths, return_probabilities=True)
        
        # Save results if requested
        if save_results:
            if output_path is None:
                output_path = "predictions.json"
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Saved predictions to {output_path}")
        
        # Log summary
        class_counts = {}
        critical_count = 0
        
        for result in results:
            pred_class = result['predicted_class']
            class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
            
            if result.get('critical_analysis', {}).get('is_critical', False):
                critical_count += 1
        
        logger.info("Prediction summary:")
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {class_name}: {count}")
        
        if critical_count > 0:
            logger.warning(f"Found {critical_count} images with critical conditions!")
        
        return results