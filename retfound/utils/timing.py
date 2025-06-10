"""
Timing Utilities
===============

Performance timing and profiling utilities.
"""

import time
import logging
from typing import Optional, Dict, Any, Callable
from contextlib import contextmanager
from functools import wraps
import numpy as np

logger = logging.getLogger(__name__)


class Timer:
    """
    Simple timer class for measuring execution time
    """
    
    def __init__(self, name: Optional[str] = None, verbose: bool = True):
        """
        Initialize timer
        
        Args:
            name: Timer name for logging
            verbose: Whether to print timing info
        """
        self.name = name or "Timer"
        self.verbose = verbose
        self.start_time = None
        self.elapsed_time = None
    
    def start(self) -> 'Timer':
        """Start the timer"""
        self.start_time = time.time()
        return self
    
    def stop(self) -> float:
        """
        Stop the timer
        
        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            raise RuntimeError("Timer not started")
        
        self.elapsed_time = time.time() - self.start_time
        
        if self.verbose:
            logger.info(f"{self.name}: {self.elapsed_time:.3f}s")
        
        return self.elapsed_time
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()


@contextmanager
def TimeIt(name: str = "Operation", logger_func: Optional[Callable] = None):
    """
    Context manager for timing code blocks
    
    Args:
        name: Name of the operation
        logger_func: Optional logging function (defaults to logger.info)
        
    Example:
        with TimeIt("Model forward pass"):
            output = model(input)
    """
    if logger_func is None:
        logger_func = logger.info
    
    start_time = time.time()
    yield
    elapsed = time.time() - start_time
    
    logger_func(f"{name} took {elapsed:.3f}s")


def timeit(func: Callable) -> Callable:
    """
    Decorator for timing function execution
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function
        
    Example:
        @timeit
        def my_function():
            # function code
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        
        logger.debug(f"{func.__name__} took {elapsed:.3f}s")
        
        return result
    
    return wrapper


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
        
    Examples:
        format_time(0.5) -> "0.50s"
        format_time(65) -> "1m 5s"
        format_time(3665) -> "1h 1m 5s"
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"


class ETA:
    """
    Estimated Time of Arrival calculator
    """
    
    def __init__(self, total_steps: int):
        """
        Initialize ETA calculator
        
        Args:
            total_steps: Total number of steps
        """
        self.total_steps = total_steps
        self.start_time = time.time()
        self.step_times = []
    
    def update(self, current_step: int) -> str:
        """
        Update ETA and return formatted string
        
        Args:
            current_step: Current step number
            
        Returns:
            ETA string
        """
        elapsed = time.time() - self.start_time
        
        if current_step > 0:
            avg_time_per_step = elapsed / current_step
            remaining_steps = self.total_steps - current_step
            eta_seconds = avg_time_per_step * remaining_steps
            
            return format_time(eta_seconds)
        
        return "Unknown"
    
    def get_progress_string(self, current_step: int) -> str:
        """
        Get formatted progress string
        
        Args:
            current_step: Current step
            
        Returns:
            Progress string with percentage and ETA
        """
        progress = current_step / self.total_steps * 100
        eta = self.update(current_step)
        elapsed = format_time(time.time() - self.start_time)
        
        return f"Progress: {progress:.1f}% | Elapsed: {elapsed} | ETA: {eta}"


class Profiler:
    """
    Simple profiler for tracking execution times of different code sections
    """
    
    def __init__(self, name: str = "Profiler"):
        """
        Initialize profiler
        
        Args:
            name: Profiler name
        """
        self.name = name
        self.timings: Dict[str, list] = {}
        self.current_timers: Dict[str, float] = {}
    
    def start(self, section: str) -> None:
        """Start timing a section"""
        self.current_timers[section] = time.time()
    
    def stop(self, section: str) -> float:
        """
        Stop timing a section
        
        Args:
            section: Section name
            
        Returns:
            Elapsed time
        """
        if section not in self.current_timers:
            raise ValueError(f"Section '{section}' not started")
        
        elapsed = time.time() - self.current_timers[section]
        
        if section not in self.timings:
            self.timings[section] = []
        
        self.timings[section].append(elapsed)
        del self.current_timers[section]
        
        return elapsed
    
    @contextmanager
    def profile(self, section: str):
        """
        Context manager for profiling a section
        
        Args:
            section: Section name
            
        Example:
            with profiler.profile("data_loading"):
                # code to profile
        """
        self.start(section)
        yield
        self.stop(section)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get profiling summary
        
        Returns:
            Dictionary with timing statistics per section
        """
        summary = {}
        
        for section, times in self.timings.items():
            times_array = np.array(times)
            summary[section] = {
                'count': len(times),
                'total': float(np.sum(times_array)),
                'mean': float(np.mean(times_array)),
                'std': float(np.std(times_array)),
                'min': float(np.min(times_array)),
                'max': float(np.max(times_array))
            }
        
        return summary
    
    def print_summary(self) -> None:
        """Print formatted profiling summary"""
        summary = self.get_summary()
        
        logger.info(f"\n{self.name} Summary")
        logger.info("=" * 80)
        logger.info(f"{'Section':<30} {'Count':>8} {'Total':>12} {'Mean':>12} {'Std':>12}")
        logger.info("-" * 80)
        
        # Sort by total time
        sorted_sections = sorted(summary.items(), key=lambda x: x[1]['total'], reverse=True)
        
        for section, stats in sorted_sections:
            logger.info(
                f"{section:<30} {stats['count']:>8} "
                f"{format_time(stats['total']):>12} "
                f"{format_time(stats['mean']):>12} "
                f"{format_time(stats['std']):>12}"
            )
        
        logger.info("=" * 80)
    
    def reset(self) -> None:
        """Reset all timings"""
        self.timings.clear()
        self.current_timers.clear()


def profile_function(func: Callable, n_runs: int = 100) -> Dict[str, float]:
    """
    Profile a function over multiple runs
    
    Args:
        func: Function to profile
        n_runs: Number of runs
        
    Returns:
        Timing statistics
    """
    times = []
    
    for _ in range(n_runs):
        start = time.time()
        func()
        elapsed = time.time() - start
        times.append(elapsed)
    
    times_array = np.array(times)
    
    return {
        'n_runs': n_runs,
        'total': float(np.sum(times_array)),
        'mean': float(np.mean(times_array)),
        'std': float(np.std(times_array)),
        'min': float(np.min(times_array)),
        'max': float(np.max(times_array)),
        'median': float(np.median(times_array))
    }


class ProgressTimer:
    """
    Timer that tracks progress and provides detailed timing statistics
    """
    
    def __init__(self, total_steps: int, update_frequency: int = 100):
        """
        Initialize progress timer
        
        Args:
            total_steps: Total number of steps
            update_frequency: How often to update statistics
        """
        self.total_steps = total_steps
        self.update_frequency = update_frequency
        self.start_time = time.time()
        self.step_times = []
        self.last_update = 0
    
    def update(self, current_step: int) -> Optional[Dict[str, Any]]:
        """
        Update progress and return statistics if update frequency reached
        
        Args:
            current_step: Current step
            
        Returns:
            Statistics dict or None
        """
        current_time = time.time()
        
        if current_step > self.last_update:
            step_time = (current_time - self.start_time) / current_step
            self.step_times.append(step_time)
        
        if current_step % self.update_frequency == 0 or current_step == self.total_steps:
            elapsed = current_time - self.start_time
            progress = current_step / self.total_steps
            
            # Calculate statistics
            recent_times = self.step_times[-self.update_frequency:]
            avg_step_time = np.mean(recent_times) if recent_times else 0
            
            eta_seconds = avg_step_time * (self.total_steps - current_step)
            
            stats = {
                'current_step': current_step,
                'total_steps': self.total_steps,
                'progress': progress,
                'elapsed': elapsed,
                'elapsed_formatted': format_time(elapsed),
                'eta': eta_seconds,
                'eta_formatted': format_time(eta_seconds),
                'avg_step_time': avg_step_time,
                'steps_per_second': 1 / avg_step_time if avg_step_time > 0 else 0
            }
            
            self.last_update = current_step
            return stats
        
        return None