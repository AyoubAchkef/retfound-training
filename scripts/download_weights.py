#!/usr/bin/env python3
"""
RETFound Weights Downloader
===========================

Download pre-trained RETFound model weights from official sources.
"""

import os
import sys
import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, List
import urllib.request
import urllib.error
from tqdm import tqdm
import logging
from datetime import datetime
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Model configurations
RETFOUND_MODELS = {
    'cfp': {
        'name': 'RETFound MAE Nature CFP',
        'filename': 'RETFound_mae_natureCFP.pth',
        'url': 'https://github.com/rmaphoh/RETFound_MAE/releases/download/v1.0/RETFound_mae_natureCFP.pth',
        'size_mb': 1320,
        'md5': None,  # Add actual MD5 if available
        'description': 'Pre-trained on color fundus photography images'
    },
    'oct': {
        'name': 'RETFound MAE Nature OCT',
        'filename': 'RETFound_mae_natureOCT.pth',
        'url': 'https://github.com/rmaphoh/RETFound_MAE/releases/download/v1.0/RETFound_mae_natureOCT.pth',
        'size_mb': 1320,
        'md5': None,
        'description': 'Pre-trained on OCT images'
    },
    'meh': {
        'name': 'RETFound MAE MEH',
        'filename': 'RETFound_mae_meh.pth',
        'url': 'https://github.com/rmaphoh/RETFound_MAE/releases/download/v1.0/RETFound_mae_meh.pth',
        'size_mb': 1320,
        'md5': None,
        'description': 'Multi-ethnic hybrid pre-training'
    }
}


class DownloadProgress:
    """Progress bar for downloads"""
    
    def __init__(self, total_size: int, desc: str = "Downloading"):
        self.pbar = tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            desc=desc,
            ncols=100
        )
    
    def __call__(self, block_num: int, block_size: int, total_size: int):
        if self.pbar.total != total_size:
            self.pbar.reset(total=total_size)
        downloaded = block_num * block_size
        self.pbar.update(downloaded - self.pbar.n)
    
    def close(self):
        self.pbar.close()


def get_file_size(url: str) -> Optional[int]:
    """Get file size from URL"""
    try:
        response = requests.head(url, allow_redirects=True, timeout=10)
        if 'content-length' in response.headers:
            return int(response.headers['content-length'])
    except Exception as e:
        logger.warning(f"Could not get file size: {e}")
    return None


def calculate_md5(filepath: Path, chunk_size: int = 8192) -> str:
    """Calculate MD5 hash of a file"""
    md5_hash = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def verify_download(filepath: Path, expected_md5: Optional[str] = None, 
                   expected_size: Optional[int] = None) -> bool:
    """Verify downloaded file"""
    if not filepath.exists():
        return False
    
    # Check file size
    if expected_size:
        actual_size = filepath.stat().st_size
        if abs(actual_size - expected_size) > 1024 * 1024:  # 1MB tolerance
            logger.error(f"Size mismatch: expected {expected_size}, got {actual_size}")
            return False
    
    # Check MD5 if provided
    if expected_md5:
        actual_md5 = calculate_md5(filepath)
        if actual_md5 != expected_md5:
            logger.error(f"MD5 mismatch: expected {expected_md5}, got {actual_md5}")
            return False
    
    return True


def download_file(url: str, filepath: Path, desc: str = "Downloading", 
                 resume: bool = True) -> bool:
    """Download file with progress bar and resume support"""
    
    # Create parent directory
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file exists and resume is enabled
    if filepath.exists() and resume:
        existing_size = filepath.stat().st_size
        headers = {'Range': f'bytes={existing_size}-'}
        mode = 'ab'
        logger.info(f"Resuming download from {existing_size} bytes")
    else:
        headers = {}
        mode = 'wb'
        existing_size = 0
    
    try:
        # Get total file size
        total_size = get_file_size(url)
        if total_size:
            desc = f"{desc} ({total_size / 1024 / 1024:.1f} MB)"
        
        # Download with progress bar
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        
        # Calculate total size for progress bar
        if total_size:
            pbar_total = total_size
        else:
            pbar_total = int(response.headers.get('content-length', 0)) + existing_size
        
        with open(filepath, mode) as f:
            with tqdm(
                total=pbar_total,
                initial=existing_size,
                unit='B',
                unit_scale=True,
                desc=desc,
                ncols=100
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        logger.info(f"Downloaded: {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        if filepath.exists() and not resume:
            filepath.unlink()  # Remove partial download
        return False


def download_weights(
    models: List[str],
    output_dir: Path,
    resume: bool = True,
    verify: bool = True,
    parallel: bool = False
) -> Dict[str, bool]:
    """Download multiple model weights"""
    
    results = {}
    
    if parallel and len(models) > 1:
        # Parallel download
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            
            for model_key in models:
                if model_key not in RETFOUND_MODELS:
                    logger.error(f"Unknown model: {model_key}")
                    results[model_key] = False
                    continue
                
                model_info = RETFOUND_MODELS[model_key]
                filepath = output_dir / model_info['filename']
                
                future = executor.submit(
                    download_single_weight,
                    model_key,
                    model_info,
                    filepath,
                    resume,
                    verify
                )
                futures[future] = model_key
            
            for future in as_completed(futures):
                model_key = futures[future]
                try:
                    results[model_key] = future.result()
                except Exception as e:
                    logger.error(f"Error downloading {model_key}: {e}")
                    results[model_key] = False
    else:
        # Sequential download
        for model_key in models:
            if model_key not in RETFOUND_MODELS:
                logger.error(f"Unknown model: {model_key}")
                results[model_key] = False
                continue
            
            model_info = RETFOUND_MODELS[model_key]
            filepath = output_dir / model_info['filename']
            
            results[model_key] = download_single_weight(
                model_key, model_info, filepath, resume, verify
            )
    
    return results


def download_single_weight(
    model_key: str,
    model_info: Dict,
    filepath: Path,
    resume: bool = True,
    verify: bool = True
) -> bool:
    """Download a single model weight file"""
    
    logger.info(f"\nDownloading {model_info['name']}...")
    logger.info(f"Description: {model_info['description']}")
    
    # Check if already exists
    if filepath.exists():
        if verify:
            logger.info("File exists, verifying...")
            expected_size = model_info.get('size_mb', 0) * 1024 * 1024
            if verify_download(filepath, model_info.get('md5'), expected_size):
                logger.info("✓ File verified successfully")
                return True
            else:
                logger.warning("✗ Verification failed, re-downloading...")
                filepath.unlink()
        else:
            logger.info("✓ File already exists")
            return True
    
    # Download file
    success = download_file(
        model_info['url'],
        filepath,
        desc=f"{model_key.upper()}",
        resume=resume
    )
    
    if success and verify:
        logger.info("Verifying download...")
        expected_size = model_info.get('size_mb', 0) * 1024 * 1024
        if verify_download(filepath, model_info.get('md5'), expected_size):
            logger.info("✓ Verification successful")
        else:
            logger.error("✗ Verification failed")
            return False
    
    return success


def list_models():
    """List available models"""
    print("\nAvailable RETFound Models:")
    print("-" * 80)
    print(f"{'Key':<10} {'Name':<30} {'Size (MB)':<10} {'Description':<30}")
    print("-" * 80)
    
    for key, info in RETFOUND_MODELS.items():
        print(f"{key:<10} {info['name']:<30} {info['size_mb']:<10} {info['description']:<30}")
    
    print("-" * 80)
    print(f"\nTotal models: {len(RETFOUND_MODELS)}")


def create_metadata(output_dir: Path, downloaded_models: Dict[str, bool]):
    """Create metadata file for downloaded models"""
    metadata = {
        'download_date': datetime.now().isoformat(),
        'models': {}
    }
    
    for model_key, success in downloaded_models.items():
        if success:
            model_info = RETFOUND_MODELS[model_key]
            filepath = output_dir / model_info['filename']
            
            metadata['models'][model_key] = {
                'filename': model_info['filename'],
                'name': model_info['name'],
                'description': model_info['description'],
                'size_bytes': filepath.stat().st_size if filepath.exists() else 0,
                'md5': calculate_md5(filepath) if filepath.exists() else None,
                'download_url': model_info['url']
            }
    
    metadata_file = output_dir / 'models_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"\nMetadata saved to: {metadata_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Download RETFound pre-trained weights',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all models
  python download_weights.py --all
  
  # Download specific models
  python download_weights.py --models cfp oct
  
  # Download to specific directory
  python download_weights.py --all --output weights/retfound
  
  # List available models
  python download_weights.py --list
        """
    )
    
    parser.add_argument(
        '--models', 
        nargs='+', 
        choices=list(RETFOUND_MODELS.keys()),
        help='Specific models to download'
    )
    parser.add_argument(
        '--all', 
        action='store_true',
        help='Download all available models'
    )
    parser.add_argument(
        '--output', 
        type=Path, 
        default=Path('weights'),
        help='Output directory for weights (default: weights)'
    )
    parser.add_argument(
        '--list', 
        action='store_true',
        help='List available models'
    )
    parser.add_argument(
        '--no-resume', 
        action='store_true',
        help='Disable resume for partial downloads'
    )
    parser.add_argument(
        '--no-verify', 
        action='store_true',
        help='Skip verification after download'
    )
    parser.add_argument(
        '--parallel', 
        action='store_true',
        help='Download multiple files in parallel'
    )
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list:
        list_models()
        return 0
    
    # Determine which models to download
    if args.all:
        models_to_download = list(RETFOUND_MODELS.keys())
    elif args.models:
        models_to_download = args.models
    else:
        parser.error('Please specify --models or --all')
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Models to download: {', '.join(models_to_download)}")
    
    # Download weights
    results = download_weights(
        models_to_download,
        args.output,
        resume=not args.no_resume,
        verify=not args.no_verify,
        parallel=args.parallel
    )
    
    # Create metadata
    create_metadata(args.output, results)
    
    # Summary
    print("\n" + "=" * 80)
    print("Download Summary:")
    print("=" * 80)
    
    success_count = sum(1 for success in results.values() if success)
    failed = [model for model, success in results.items() if not success]
    
    print(f"✓ Successfully downloaded: {success_count}/{len(results)}")
    
    if failed:
        print(f"✗ Failed downloads: {', '.join(failed)}")
        return 1
    
    print("\nAll downloads completed successfully!")
    print(f"Weights saved to: {args.output.absolute()}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())