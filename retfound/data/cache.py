"""
Caching System for RETFound
==========================

Implements efficient caching mechanisms for datasets and images
to speed up data loading during training.
"""

import os
import shutil
import hashlib
import pickle
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
from datetime import datetime, timedelta
import threading
from functools import lru_cache

import numpy as np
from PIL import Image

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
            'created_at': datetime.now().isoformat()
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
            
            # Reset metadata
            self.metadata = {
                'entries': {},
                'total_size': 0,
                'created_at': datetime.now().isoformat()
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
            'cache_dir': str(self.cache_dir)
        }


class DatasetCache(CacheManager):
    """Cache manager specifically for dataset metadata and processed data"""
    
    def __init__(self, cache_dir: Union[str, Path], **kwargs):
        """Initialize dataset cache"""
        super().__init__(cache_dir, **kwargs)
        
        # Additional dataset-specific caching
        self.dataset_info_cache = {}
    
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
                    'type': 'dataset_info'
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


class ImageCache(CacheManager):
    """Cache manager for preprocessed images"""
    
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
    
    def _get_cache_key(self, image_path: Union[str, Path]) -> str:
        """Generate cache key for an image path"""
        image_path = str(image_path)
        return hashlib.md5(image_path.encode()).hexdigest()
    
    def cache_image(
        self,
        image_path: Union[str, Path],
        image_array: np.ndarray
    ) -> bool:
        """
        Cache a preprocessed image
        
        Args:
            image_path: Original image path
            image_array: Preprocessed image array
            
        Returns:
            Success status
        """
        key = self._get_cache_key(image_path)
        cache_file = self.cache_dir / f"{key}.pkl"
        
        with self.lock:
            try:
                # Save to disk
                with open(cache_file, 'wb') as f:
                    if self.enable_compression:
                        import gzip
                        f = gzip.open(cache_file.with_suffix('.pkl.gz'), 'wb')
                    pickle.dump(image_array, f)
                
                # Update metadata
                file_size = cache_file.stat().st_size
                self.metadata['entries'][key] = {
                    'filename': cache_file.name,
                    'size': file_size,
                    'created_at': datetime.now().isoformat(),
                    'type': 'image',
                    'original_path': str(image_path),
                    'shape': image_array.shape
                }
                self.metadata['total_size'] += file_size
                
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
    ) -> Optional[np.ndarray]:
        """
        Get cached image
        
        Args:
            image_path: Original image path
            
        Returns:
            Cached image array or None
        """
        key = self._get_cache_key(image_path)
        
        # Check memory cache first
        if key in self._memory_cache:
            self._access_count[key] = self._access_count.get(key, 0) + 1
            return self._memory_cache[key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{key}.pkl"
        if self.enable_compression:
            cache_file = cache_file.with_suffix('.pkl.gz')
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    if self.enable_compression:
                        import gzip
                        f = gzip.open(cache_file, 'rb')
                    image_array = pickle.load(f)
                
                # Add to memory cache
                self._add_to_memory_cache(key, image_array)
                
                return image_array
                
            except Exception as e:
                logger.error(f"Failed to load cached image: {e}")
                # Remove corrupted cache entry
                self._remove_entry(key)
        
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
    
    def preload_images(self, image_paths: List[Union[str, Path]]):
        """
        Preload multiple images into cache
        
        Args:
            image_paths: List of image paths to preload
        """
        logger.info(f"Preloading {len(image_paths)} images into cache...")
        
        loaded = 0
        for image_path in image_paths:
            if self.get_image(image_path) is not None:
                loaded += 1
        
        logger.info(f"Preloaded {loaded}/{len(image_paths)} images from cache")


# Global cache instances
_dataset_cache: Optional[DatasetCache] = None
_image_cache: Optional[ImageCache] = None


def get_dataset_cache(cache_dir: Optional[Union[str, Path]] = None) -> DatasetCache:
    """Get global dataset cache instance"""
    global _dataset_cache
    
    if _dataset_cache is None:
        if cache_dir is None:
            cache_dir = Path.home() / '.cache' / 'retfound' / 'datasets'
        _dataset_cache = DatasetCache(cache_dir)
    
    return _dataset_cache


def get_image_cache(cache_dir: Optional[Union[str, Path]] = None) -> ImageCache:
    """Get global image cache instance"""
    global _image_cache
    
    if _image_cache is None:
        if cache_dir is None:
            cache_dir = Path.home() / '.cache' / 'retfound' / 'images'
        _image_cache = ImageCache(cache_dir)
    
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


def get_cache_stats() -> Dict[str, Any]:
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
    except Exception as e:
        stats['image_cache'] = {'error': str(e)}
    
    return stats
