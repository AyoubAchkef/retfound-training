"""
I/O Utilities
============

File I/O and path utilities.
"""

import json
import yaml
import shutil
import logging
from pathlib import Path
from typing import Any, Dict, Union, Optional

logger = logging.getLogger(__name__)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON file
    
    Args:
        path: Path to JSON file
        
    Returns:
        Loaded data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    logger.debug(f"Loaded JSON from {path}")
    return data


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to JSON file
    
    Args:
        data: Data to save
        path: Path to save to
        indent: JSON indentation
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)
    
    logger.debug(f"Saved JSON to {path}")


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML file
    
    Args:
        path: Path to YAML file
        
    Returns:
        Loaded data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If file is not valid YAML
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    
    logger.debug(f"Loaded YAML from {path}")
    return data


def save_yaml(data: Any, path: Union[str, Path], default_flow_style: bool = False) -> None:
    """
    Save data to YAML file
    
    Args:
        data: Data to save
        path: Path to save to
        default_flow_style: YAML flow style
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=default_flow_style)
    
    logger.debug(f"Saved YAML to {path}")


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if not
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_size(path: Union[str, Path], unit: str = 'MB') -> float:
    """
    Get file size in specified unit
    
    Args:
        path: File path
        unit: Size unit (B, KB, MB, GB)
        
    Returns:
        File size in specified unit
    """
    path = Path(path)
    if not path.exists():
        return 0.0
    
    size_bytes = path.stat().st_size
    
    units = {
        'B': 1,
        'KB': 1024,
        'MB': 1024 ** 2,
        'GB': 1024 ** 3
    }
    
    if unit not in units:
        raise ValueError(f"Unknown unit: {unit}")
    
    return size_bytes / units[unit]


def copy_file(src: Union[str, Path], dst: Union[str, Path], overwrite: bool = False) -> None:
    """
    Copy file from source to destination
    
    Args:
        src: Source file path
        dst: Destination file path
        overwrite: Whether to overwrite if destination exists
        
    Raises:
        FileNotFoundError: If source doesn't exist
        FileExistsError: If destination exists and overwrite is False
    """
    src = Path(src)
    dst = Path(dst)
    
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")
    
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {dst}")
    
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    
    logger.debug(f"Copied {src} to {dst}")


def move_file(src: Union[str, Path], dst: Union[str, Path], overwrite: bool = False) -> None:
    """
    Move file from source to destination
    
    Args:
        src: Source file path
        dst: Destination file path
        overwrite: Whether to overwrite if destination exists
        
    Raises:
        FileNotFoundError: If source doesn't exist
        FileExistsError: If destination exists and overwrite is False
    """
    src = Path(src)
    dst = Path(dst)
    
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")
    
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {dst}")
    
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    
    logger.debug(f"Moved {src} to {dst}")


def list_files(
    directory: Union[str, Path],
    pattern: str = '*',
    recursive: bool = False
) -> list:
    """
    List files in directory
    
    Args:
        directory: Directory path
        pattern: File pattern (e.g., '*.jpg')
        recursive: Whether to search recursively
        
    Returns:
        List of file paths
    """
    directory = Path(directory)
    
    if not directory.exists():
        return []
    
    if recursive:
        files = list(directory.rglob(pattern))
    else:
        files = list(directory.glob(pattern))
    
    return sorted(files)


def read_text(path: Union[str, Path], encoding: str = 'utf-8') -> str:
    """
    Read text file
    
    Args:
        path: File path
        encoding: File encoding
        
    Returns:
        File contents as string
    """
    path = Path(path)
    with open(path, 'r', encoding=encoding) as f:
        return f.read()


def write_text(text: str, path: Union[str, Path], encoding: str = 'utf-8') -> None:
    """
    Write text to file
    
    Args:
        text: Text to write
        path: File path
        encoding: File encoding
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding=encoding) as f:
        f.write(text)


def atomic_save(data: Any, path: Union[str, Path], save_fn: callable) -> None:
    """
    Atomically save data to file (write to temp file then rename)
    
    Args:
        data: Data to save
        path: Target file path
        save_fn: Function to save data (e.g., torch.save, json.dump)
    """
    path = Path(path)
    temp_path = path.with_suffix('.tmp')
    
    try:
        # Save to temporary file
        save_fn(data, temp_path)
        
        # Rename to final path (atomic on most systems)
        temp_path.replace(path)
        
        logger.debug(f"Atomically saved to {path}")
        
    except Exception as e:
        # Clean up temp file on error
        if temp_path.exists():
            temp_path.unlink()
        raise e