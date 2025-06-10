"""
Caching System for RETFound - Enhanced for Dataset v6.1
=======================================================

Implements efficient caching mechanisms for datasets and images
to speed up data loading during training, with specific optimizations
for the CAASI dataset v6.1 structure.
"""

import os
import shutil
import hashlib
import pickle
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union, List, Tuple
from datetime import datetime, timedelta
import threading
from functools import lru_cache
from collections import defaultdict

import numpy as np
from PIL import Image
import torch

logger = logging.getLogger(__name__)


class CacheManager:
    """Base class for cache management"""
    
    def __init__(
        self,
        cache_dir: Union[str, Path],
        max_size_gb: float = 50.0,
        ttl_days: int = 30,
        enable_compression: bool = False
    ):
        """
        Initialize cache manager
        
        Args:
            cache_dir: Directory for cache storage
            max_size_gb: Maximum cache size in GB
            ttl_days: Time to live for cache entries in days
            enable_compression: Whether to compress cached data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        self.ttl = timedelta(days=ttl_days)
        self.enable_compression = enable_compression
        
        # Cache metadata
        self.metadata_file = self.cache_dir / 'cache_metadata.json'
        self.metadata = self._load_metadata()
        
        # Thread lock for concurrent access
        self.lock = threading.Lock()
        
        # Clean up old entries on initialization
        self._cleanup_old_entries()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
        
        return {
            'entries': {},
            'total_size': 0,
            'created_at': datetime.now().isoformat(),
            'version': 'v6.1'  # Track dataset version
        }
    
    def _save_metadata(self):
        """Save cache metadata"""
        with self.lock:
            try:
                with open(self.metadata_file, 'w') as f:
                    json.dump(self.metadata, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save cache metadata: {e}")
    
    def _cleanup_old_entries(self):
        """Remove cache entries older than TTL"""
        current_time = datetime.now()
        entries_to_remove = []
        
        for key, entry_info in self.metadata['entries'].items():
            created_at = datetime.fromisoformat(entry_info['created_at'])
            if current_time - created_at > self.ttl:
                entries_to_remove.append(key)
        
        # Remove old entries
        for key in entries_to_remove:
            self._remove_entry(key)
        
        if entries_to_remove:
            logger.info(f"Cleaned up {len(entries_to_remove)} old cache entries")
    
    def _remove_entry(self, key: str):
        """Remove a cache entry"""
        if key in self.metadata['entries']:
            entry_info = self.metadata['entries'][key]
            cache_file = self.cache_dir / entry_info['filename']
            
            # Remove file
            if cache_file.exists():
                try:
                    cache_file.unlink()
                    self.metadata['total_size'] -= entry_info['size']
                except Exception as e:
                    logger.error(f"Failed to remove cache file {cache_file}: {e}")
            
            # Remove from metadata
            del self.metadata['entries'][key]
    
    def _enforce_size_limit(self):
        """Enforce maximum cache size by removing oldest entries"""
        if self.metadata['total_size'] <= self.max_size_bytes:
            return
        
        # Sort entries by creation time
        sorted_entries = sorted(
            self.metadata['entries'].items(),
            key=lambda x: x[1]['created_at']
        )
        
        # Remove oldest entries until under limit
        while self.metadata['total_size'] > self.max_size_bytes and sorted_entries:
            key, _ = sorted_entries.pop(0)
            self._remove_entry(key)
        
        logger.info(f"Cache size enforced: {self.metadata['total_size'] / 1e9:.1f}GB")
    
    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            # Remove all cache files
            for cache_file in self.cache_dir.glob('*.pkl'):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.error(f"Failed to remove {cache_file}: {e}")
            
            # Also remove compressed files
            for cache_file in self.cache_dir.glob('*.pkl.gz'):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.error(f"Failed to remove {cache_file}: {e}")
            
            # Reset metadata
            self.metadata = {
                'entries': {},
                'total_size': 0,
                'created_at': datetime.now().isoformat(),
                'version': 'v6.1'
            }
            self._save_metadata()
        
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'num_entries': len(self.metadata['entries']),
            'total_size_gb': self.metadata['total_size'] / 1e9,
            'max_size_gb': self.max_size_bytes / 1e9,
            'usage_percent': (self.metadata['total_size'] / self.max_size_bytes) * 100,
            'created_at': self.metadata['created_at'],
            'cache_dir': str(self.cache_dir),
            'version': self.metadata.get('version', 'unknown')
        }


class DatasetCache(CacheManager):
    """Cache manager specifically for dataset metadata and processed data"""
    
    def __init__(self, cache_dir: Union[str, Path], **kwargs):
        """Initialize dataset cache"""
        super().__init__(cache_dir, **kwargs)
        
        # Additional dataset-specific caching
        self.dataset_info_cache = {}
        
        # V6.1 specific caches
        self.modality_stats_cache = {}  # Stats per modality
        self.class_distribution_cache = {}  # Class distribution cache
        self.unified_class_names_cache = None  # 28 unified classes
    
    def cache_dataset_info(
        self,
        dataset_id: str,
        info: Dict[str, Any]
    ):
        """
        Cache dataset information
        
        Args:
            dataset_id: Unique dataset identifier
            info: Dataset information to cache
        """
        key = f"dataset_info_{dataset_id}"
        cache_file = self.cache_dir / f"{key}.json"
        
        with self.lock:
            try:
                with open(cache_file, 'w') as f:
                    json.dump(info, f, indent=2)
                
                # Update metadata
                self.metadata['entries'][key] = {
                    'filename': cache_file.name,
                    'size': cache_file.stat().st_size,
                    'created_at': datetime.now().isoformat(),
                    'type': 'dataset_info',
                    'dataset_version': info.get('version', 'unknown')
                }
                self.metadata['total_size'] += cache_file.stat().st_size
                self._save_metadata()
                
                # Update in-memory cache
                self.dataset_info_cache[dataset_id] = info
                
            except Exception as e:
                logger.error(f"Failed to cache dataset info: {e}")
    
    def get_dataset_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get cached dataset information"""
        # Check in-memory cache first
        if dataset_id in self.dataset_info_cache:
            return self.dataset_info_cache[dataset_id]
        
        # Check disk cache
        key = f"dataset_info_{dataset_id}"
        cache_file = self.cache_dir / f"{key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    info = json.load(f)
                
                # Update in-memory cache
                self.dataset_info_cache[dataset_id] = info
                return info
                
            except Exception as e:
                logger.error(f"Failed to load cached dataset info: {e}")
        
        return None
    
    def cache_splits(
        self,
        dataset_id: str,
        splits: Dict[str, List[int]]
    ):
        """Cache dataset splits (train/val/test indices)"""
        key = f"splits_{dataset_id}"
        cache_file = self.cache_dir / f"{key}.pkl"
        
        with self.lock:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(splits, f)
                
                # Update metadata
                self.metadata['entries'][key] = {
                    'filename': cache_file.name,
                    'size': cache_file.stat().st_size,
                    'created_at': datetime.now().isoformat(),
                    'type': 'splits'
                }
                self.metadata['total_size'] += cache_file.stat().st_size
                self._save_metadata()
                
            except Exception as e:
                logger.error(f"Failed to cache splits: {e}")
    
    def get_splits(self, dataset_id: str) -> Optional[Dict[str, List[int]]]:
        """Get cached dataset splits"""
        key = f"splits_{dataset_id}"
        cache_file = self.cache_dir / f"{key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Failed to load cached splits: {e}")
        
        return None
    
    # V6.1 specific methods
    def cache_v61_metadata(
        self,
        metadata: Dict[str, Any]
    ):
        """Cache dataset v6.1 specific metadata"""
        key = "v61_metadata"
        cache_file = self.cache_dir / f"{key}.json"
        
        with self.lock:
            try:
                # Ensure metadata includes v6.1 specifics
                metadata['dataset_version'] = 'v6.1'
                metadata['num_classes'] = 28
                metadata['modalities'] = ['fundus', 'oct']
                metadata['cached_at'] = datetime.now().isoformat()
                
                with open(cache_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Update cache metadata
                self.metadata['entries'][key] = {
                    'filename': cache_file.name,
                    'size': cache_file.stat().st_size,
                    'created_at': datetime.now().isoformat(),
                    'type': 'v61_metadata'
                }
                self.metadata['total_size'] += cache_file.stat().st_size
                self._save_metadata()
                
                logger.info("Cached v6.1 metadata successfully")
                
            except Exception as e:
                logger.error(f"Failed to cache v6.1 metadata: {e}")
    
    def get_v61_metadata(self) -> Optional[Dict[str, Any]]:
        """Get cached v6.1 metadata"""
        key = "v61_metadata"
        cache_file = self.cache_dir / f"{key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load v6.1 metadata: {e}")
        
        return None
    
    def cache_modality_stats(
        self,
        modality: str,
        stats: Dict[str, Any]
    ):
        """
        Cache statistics for a specific modality (fundus/oct)
        
        Args:
            modality: 'fundus' or 'oct'
            stats: Statistics dictionary
        """
        key = f"modality_stats_{modality}"
        cache_file = self.cache_dir / f"{key}.json"
        
        with self.lock:
            try:
                stats['modality'] = modality
                stats['cached_at'] = datetime.now().isoformat()
                
                with open(cache_file, 'w') as f:
                    json.dump(stats, f, indent=2)
                
                # Update in-memory cache
                self.modality_stats_cache[modality] = stats
                
                # Update metadata
                self.metadata['entries'][key] = {
                    'filename': cache_file.name,
                    'size': cache_file.stat().st_size,
                    'created_at': datetime.now().isoformat(),
                    'type': 'modality_stats'
                }
                self.metadata['total_size'] += cache_file.stat().st_size
                self._save_metadata()
                
            except Exception as e:
                logger.error(f"Failed to cache modality stats for {modality}: {e}")
    
    def get_modality_stats(self, modality: str) -> Optional[Dict[str, Any]]:
        """Get cached modality statistics"""
        # Check in-memory cache first
        if modality in self.modality_stats_cache:
            return self.modality_stats_cache[modality]
        
        # Check disk cache
        key = f"modality_stats_{modality}"
        cache_file = self.cache_dir / f"{key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    stats = json.load(f)
                
                # Update in-memory cache
                self.modality_stats_cache[modality] = stats
                return stats
                
            except Exception as e:
                logger.error(f"Failed to load modality stats for {modality}: {e}")
        
        return None
    
    def cache_class_distribution(
        self,
        split: str,
        distribution: Dict[str, int],
        modality: Optional[str] = None
    ):
        """
        Cache class distribution for a specific split
        
        Args:
            split: 'train', 'val', or 'test'
            distribution: Class distribution dictionary
            modality: Optional modality filter
        """
        key = f"class_dist_{split}"
        if modality:
            key += f"_{modality}"
        
        cache_file = self.cache_dir / f"{key}.json"
        
        with self.lock:
            try:
                data = {
                    'split': split,
                    'modality': modality,
                    'distribution': distribution,
                    'total_samples': sum(distribution.values()),
                    'cached_at': datetime.now().isoformat()
                }
                
                with open(cache_file, 'w') as f:
                    json.dump(data, f, indent=2)
                
                # Update in-memory cache
                cache_key = f"{split}_{modality}" if modality else split
                self.class_distribution_cache[cache_key] = data
                
                # Update metadata
                self.metadata['entries'][key] = {
                    'filename': cache_file.name,
                    'size': cache_file.stat().st_size,
                    'created_at': datetime.now().isoformat(),
                    'type': 'class_distribution'
                }
                self.metadata['total_size'] += cache_file.stat().st_size
                self._save_metadata()
                
            except Exception as e:
                logger.error(f"Failed to cache class distribution: {e}")
    
    def get_class_distribution(
        self,
        split: str,
        modality: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get cached class distribution"""
        cache_key = f"{split}_{modality}" if modality else split
        
        # Check in-memory cache first
        if cache_key in self.class_distribution_cache:
            return self.class_distribution_cache[cache_key]
        
        # Check disk cache
        key = f"class_dist_{split}"
        if modality:
            key += f"_{modality}"
        
        cache_file = self.cache_dir / f"{key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                # Update in-memory cache
                self.class_distribution_cache[cache_key] = data
                return data
                
            except Exception as e:
                logger.error(f"Failed to load class distribution: {e}")
        
        return None


class ImageCache(CacheManager):
    """Cache manager for preprocessed images with v6.1 optimizations"""
    
    def __init__(
        self,
        cache_dir: Union[str, Path],
        image_size: int = 224,
        **kwargs
    ):
        """
        Initialize image cache
        
        Args:
            cache_dir: Directory for cache storage
            image_size: Expected image size
            **kwargs: Additional arguments for CacheManager
        """
        super().__init__(cache_dir, **kwargs)
        self.image_size = image_size
        
        # LRU cache for recently accessed images
        self._memory_cache = {}
        self._memory_cache_size = 1000  # Number of images to keep in memory
        self._access_count = {}
        
        # V6.1 specific: Track modality in cache
        self._modality_cache = {}  # image_key -> modality mapping
        
        # Stats tracking
        self._cache_hits = defaultdict(int)
        self._cache_misses = defaultdict(int)
    
    def _get_cache_key(self, image_path: Union[str, Path]) -> str:
        """Generate cache key for an image path"""
        image_path = str(image_path)
        # Include image size in key for multi-resolution support
        key_string = f"{image_path}_{self.image_size}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _detect_modality(self, image_path: Union[str, Path]) -> str:
        """Detect modality from image path (fundus or oct)"""
        path_str = str(image_path).lower()
        if 'fundus' in path_str:
            return 'fundus'
        elif 'oct' in path_str:
            return 'oct'
        else:
            # Try to infer from parent directories
            parts = Path(image_path).parts
            for part in parts:
                if 'fundus' in part.lower():
                    return 'fundus'
                elif 'oct' in part.lower():
                    return 'oct'
            return 'unknown'
    
    def cache_image(
        self,
        image_path: Union[str, Path],
        image_array: np.ndarray,
        modality: Optional[str] = None
    ) -> bool:
        """
        Cache a preprocessed image
        
        Args:
            image_path: Original image path
            image_array: Preprocessed image array
            modality: Optional modality hint ('fundus' or 'oct')
            
        Returns:
            Success status
        """
        key = self._get_cache_key(image_path)
        cache_file = self.cache_dir / f"{key}.pkl"
        
        # Detect modality if not provided
        if modality is None:
            modality = self._detect_modality(image_path)
        
        with self.lock:
            try:
                # Prepare cache data
                cache_data = {
                    'array': image_array,
                    'modality': modality,
                    'shape': image_array.shape,
                    'dtype': str(image_array.dtype),
                    'path': str(image_path),
                    'cached_at': datetime.now().isoformat()
                }
                
                # Save to disk
                if self.enable_compression:
                    import gzip
                    cache_file = cache_file.with_suffix('.pkl.gz')
                    with gzip.open(cache_file, 'wb') as f:
                        pickle.dump(cache_data, f)
                else:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(cache_data, f)
                
                # Update metadata
                file_size = cache_file.stat().st_size
                self.metadata['entries'][key] = {
                    'filename': cache_file.name,
                    'size': file_size,
                    'created_at': datetime.now().isoformat(),
                    'type': 'image',
                    'original_path': str(image_path),
                    'shape': image_array.shape,
                    'modality': modality
                }
                self.metadata['total_size'] += file_size
                
                # Track modality
                self._modality_cache[key] = modality
                
                # Enforce size limit
                self._enforce_size_limit()
                self._save_metadata()
                
                # Add to memory cache
                self._add_to_memory_cache(key, image_array)
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to cache image {image_path}: {e}")
                return False
    
    def get_image(
        self,
        image_path: Union[str, Path]
    ) -> Optional[Union[np.ndarray, Tuple[np.ndarray, str]]]:
        """
        Get cached image
        
        Args:
            image_path: Original image path
            
        Returns:
            Cached image array or None
        """
        key = self._get_cache_key(image_path)
        modality = self._detect_modality(image_path)
        
        # Check memory cache first
        if key in self._memory_cache:
            self._access_count[key] = self._access_count.get(key, 0) + 1
            self._cache_hits[modality] += 1
            return self._memory_cache[key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{key}.pkl"
        if self.enable_compression:
            cache_file = cache_file.with_suffix('.pkl.gz')
        
        if cache_file.exists():
            try:
                if self.enable_compression:
                    import gzip
                    with gzip.open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                else:
                    with open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                
                # Extract image array
                if isinstance(cache_data, dict):
                    image_array = cache_data['array']
                    # Update modality cache
                    self._modality_cache[key] = cache_data.get('modality', modality)
                else:
                    # Legacy format compatibility
                    image_array = cache_data
                
                # Add to memory cache
                self._add_to_memory_cache(key, image_array)
                
                self._cache_hits[modality] += 1
                return image_array
                
            except Exception as e:
                logger.error(f"Failed to load cached image: {e}")
                # Remove corrupted cache entry
                self._remove_entry(key)
        
        self._cache_misses[modality] += 1
        return None
    
    def _add_to_memory_cache(self, key: str, image_array: np.ndarray):
        """Add image to memory cache with LRU eviction"""
        # If cache is full, remove least recently used
        if len(self._memory_cache) >= self._memory_cache_size:
            # Find least accessed key
            if self._access_count:
                lru_key = min(self._access_count, key=self._access_count.get)
                del self._memory_cache[lru_key]
                del self._access_count[lru_key]
        
        self._memory_cache[key] = image_array
        self._access_count[key] = 1
    
    def preload_images(
        self,
        image_paths: List[Union[str, Path]],
        show_progress: bool = True
    ):
        """
        Preload multiple images into cache
        
        Args:
            image_paths: List of image paths to preload
            show_progress: Whether to show progress bar
        """
        logger.info(f"Preloading {len(image_paths)} images into cache...")
        
        if show_progress:
            try:
                from tqdm import tqdm
                image_paths = tqdm(image_paths, desc="Preloading images")
            except ImportError:
                pass
        
        loaded = 0
        for image_path in image_paths:
            if self.get_image(image_path) is not None:
                loaded += 1
        
        logger.info(f"Preloaded {loaded}/{len(image_paths)} images from cache")
    
    def get_cache_stats_by_modality(self) -> Dict[str, Dict[str, Any]]:
        """Get cache statistics broken down by modality"""
        stats = {
            'fundus': {
                'hits': self._cache_hits['fundus'],
                'misses': self._cache_misses['fundus'],
                'hit_rate': self._cache_hits['fundus'] / max(1, self._cache_hits['fundus'] + self._cache_misses['fundus']),
                'cached_images': sum(1 for v in self._modality_cache.values() if v == 'fundus')
            },
            'oct': {
                'hits': self._cache_hits['oct'],
                'misses': self._cache_misses['oct'],
                'hit_rate': self._cache_hits['oct'] / max(1, self._cache_hits['oct'] + self._cache_misses['oct']),
                'cached_images': sum(1 for v in self._modality_cache.values() if v == 'oct')
            },
            'overall': {
                'total_hits': sum(self._cache_hits.values()),
                'total_misses': sum(self._cache_misses.values()),
                'memory_cache_size': len(self._memory_cache),
                'disk_cache_size_gb': self.metadata['total_size'] / 1e9
            }
        }
        
        return stats


# Global cache instances
_dataset_cache: Optional[DatasetCache] = None
_image_cache: Optional[ImageCache] = None


def get_dataset_cache(cache_dir: Optional[Union[str, Path]] = None) -> DatasetCache:
    """Get global dataset cache instance"""
    global _dataset_cache
    
    if _dataset_cache is None:
        if cache_dir is None:
            cache_dir = Path.home() / '.cache' / 'retfound' / 'datasets_v61'
        _dataset_cache = DatasetCache(cache_dir)
    
    return _dataset_cache


def get_image_cache(
    cache_dir: Optional[Union[str, Path]] = None,
    image_size: int = 224
) -> ImageCache:
    """Get global image cache instance"""
    global _image_cache
    
    if _image_cache is None:
        if cache_dir is None:
            cache_dir = Path.home() / '.cache' / 'retfound' / 'images_v61'
        _image_cache = ImageCache(cache_dir, image_size=image_size)
    
    return _image_cache


def clear_cache(cache_type: Optional[str] = None):
    """
    Clear cache
    
    Args:
        cache_type: Type of cache to clear ('dataset', 'image', or None for all)
    """
    if cache_type in [None, 'dataset']:
        dataset_cache = get_dataset_cache()
        dataset_cache.clear()
        logger.info("Dataset cache cleared")
    
    if cache_type in [None, 'image']:
        image_cache = get_image_cache()
        image_cache.clear()
        logger.info("Image cache cleared")


def get_cache_stats(detailed: bool = False) -> Dict[str, Any]:
    """Get statistics for all caches"""
    stats = {}
    
    try:
        dataset_cache = get_dataset_cache()
        stats['dataset_cache'] = dataset_cache.get_stats()
    except Exception as e:
        stats['dataset_cache'] = {'error': str(e)}
    
    try:
        image_cache = get_image_cache()
        stats['image_cache'] = image_cache.get_stats()
        
        if detailed:
            stats['image_cache']['modality_stats'] = image_cache.get_cache_stats_by_modality()
    except Exception as e:
        stats['image_cache'] = {'error': str(e)}
    
    return stats


def cache_v61_dataset_info(dataset_path: Union[str, Path]):
    """
    Cache comprehensive dataset v6.1 information
    
    Args:
        dataset_path: Path to the dataset root
    """
    dataset_path = Path(dataset_path)
    dataset_cache = get_dataset_cache()
    
    # Compute dataset statistics
    fundus_path = dataset_path / 'fundus'
    oct_path = dataset_path / 'oct'
    
    dataset_info = {
        'version': 'v6.1',
        'root_path': str(dataset_path),
        'total_images': 211952,
        'num_classes': 28,
        'modalities': {
            'fundus': {
                'total': 44815,
                'train': 35848,
                'val': 4472,
                'test': 4495,
                'num_classes': 18
            },
            'oct': {
                'total': 167137,
                'train': 133813,
                'val': 16627,
                'test': 16697,
                'num_classes': 10
            }
        },
        'unified_classes': 28,
        'class_names': [
            # Fundus classes (0-17)
            'Fundus_Normal', 'Fundus_DR_Mild', 'Fundus_DR_Moderate', 'Fundus_DR_Severe',
            'Fundus_DR_Proliferative', 'Fundus_Glaucoma_Suspect', 'Fundus_Glaucoma_Positive',
            'Fundus_RVO', 'Fundus_RAO', 'Fundus_Hypertensive_Retinopathy', 'Fundus_Drusen',
            'Fundus_CNV_Wet_AMD', 'Fundus_Myopia_Degenerative', 'Fundus_Retinal_Detachment',
            'Fundus_Macular_Scar', 'Fundus_Cataract_Suspected', 'Fundus_Optic_Disc_Anomaly',
            'Fundus_Other',
            # OCT classes (18-27)
            'OCT_Normal', 'OCT_DME', 'OCT_CNV', 'OCT_Dry_AMD', 'OCT_ERM',
            'OCT_Vitreomacular_Interface_Disease', 'OCT_CSR', 'OCT_RVO', 'OCT_Glaucoma', 'OCT_RAO'
        ],
        'critical_conditions': ['RAO', 'RVO', 'Retinal_Detachment', 'CNV', 'DR_Proliferative'],
        'minority_classes': ['ERM', 'RVO_OCT', 'RAO_OCT', 'Myopia_Degenerative'],
        'cached_at': datetime.now().isoformat()
    }
    
    # Cache the dataset info
    dataset_cache.cache_dataset_info('caasi_v61', dataset_info)
    dataset_cache.cache_v61_metadata(dataset_info)
    
    # Cache modality-specific stats
    for modality in ['fundus', 'oct']:
        stats = dataset_info['modalities'][modality]
        dataset_cache.cache_modality_stats(modality, stats)
    
    logger.info("Cached comprehensive v6.1 dataset information")