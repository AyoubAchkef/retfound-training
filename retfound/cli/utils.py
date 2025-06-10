"""
CLI Utilities
=============

Common utilities for command line interface.
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import torch
import psutil
import platform
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.syntax import Syntax
import click

logger = logging.getLogger(__name__)


def setup_console_logging(verbose: bool = False):
    """Setup rich console logging"""
    from rich.logging import RichHandler
    
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_time=True)]
    )


def print_banner():
    """Print RETFound CLI banner"""
    console = Console()
    
    banner = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║                    🔬 RETFound CLI v2.0                   ║
║                                                           ║
║         Foundation Model for Retinal Imaging              ║
║                    CAASI Medical AI                       ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
    """
    
    console.print(Panel(banner, style="bold blue"))


def check_environment() -> Dict[str, Any]:
    """Check and display environment information"""
    console = Console()
    
    env_info = {
        'python_version': sys.version.split()[0],
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'gpu_names': [],
        'cpu_count': psutil.cpu_count(),
        'ram_gb': psutil.virtual_memory().total / (1024**3),
        'platform': platform.platform()
    }
    
    if env_info['cuda_available']:
        for i in range(env_info['gpu_count']):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            env_info['gpu_names'].append(f"{gpu_name} ({gpu_memory:.1f}GB)")
    
    # Create environment table
    table = Table(title="Environment Information", show_header=True)
    table.add_column("Component", style="cyan", no_wrap=True)
    table.add_column("Status", style="green")
    
    table.add_row("Python", env_info['python_version'])
    table.add_row("PyTorch", env_info['pytorch_version'])
    table.add_row("Platform", env_info['platform'])
    table.add_row("CPU Cores", str(env_info['cpu_count']))
    table.add_row("RAM", f"{env_info['ram_gb']:.1f} GB")
    
    if env_info['cuda_available']:
        table.add_row("CUDA", f"✅ {env_info['cuda_version']}")
        for i, gpu in enumerate(env_info['gpu_names']):
            table.add_row(f"GPU {i}", gpu)
    else:
        table.add_row("CUDA", "❌ Not available")
    
    console.print(table)
    
    # Check for issues
    issues = []
    if not env_info['cuda_available']:
        issues.append("⚠️  CUDA not available - Training will be very slow on CPU")
    
    if env_info['gpu_count'] > 0:
        for i in range(env_info['gpu_count']):
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            if gpu_memory < 16:
                issues.append(f"⚠️  GPU {i} has only {gpu_memory:.1f}GB - RETFound needs 16GB+")
    
    if env_info['ram_gb'] < 32:
        issues.append(f"⚠️  System has only {env_info['ram_gb']:.1f}GB RAM - 32GB+ recommended")
    
    if issues:
        console.print("\n[yellow]Warnings:[/yellow]")
        for issue in issues:
            console.print(f"  {issue}")
    
    return env_info


def validate_paths(paths: Dict[str, Path]) -> bool:
    """Validate required paths exist"""
    console = Console()
    all_valid = True
    
    for name, path in paths.items():
        if path and not path.exists():
            console.print(f"[red]✗[/red] {name}: {path} does not exist")
            all_valid = False
        elif path:
            console.print(f"[green]✓[/green] {name}: {path}")
    
    return all_valid


def create_progress_bar(description: str = "Processing") -> Progress:
    """Create a rich progress bar"""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=Console(),
        transient=True
    )


def format_time(seconds: float) -> str:
    """Format seconds into human readable time"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_number(num: float, precision: int = 2) -> str:
    """Format large numbers with K/M/B suffixes"""
    if abs(num) < 1000:
        return str(round(num, precision))
    elif abs(num) < 1e6:
        return f"{num/1e3:.{precision}f}K"
    elif abs(num) < 1e9:
        return f"{num/1e6:.{precision}f}M"
    else:
        return f"{num/1e9:.{precision}f}B"


def print_model_summary(model: torch.nn.Module, input_size: tuple = (3, 224, 224)):
    """Print model summary with parameter counts"""
    console = Console()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # Create summary table
    table = Table(title="Model Summary", show_header=True)
    table.add_column("Layer Type", style="cyan")
    table.add_column("Output Shape", style="green")
    table.add_column("Parameters", style="yellow", justify="right")
    
    # Add model info
    console.print(f"\n[bold]Model Architecture:[/bold] {model.__class__.__name__}")
    console.print(f"[bold]Total Parameters:[/bold] {format_number(total_params)} ({total_params:,})")
    console.print(f"[bold]Trainable:[/bold] {format_number(trainable_params)} ({trainable_params:,})")
    console.print(f"[bold]Non-trainable:[/bold] {format_number(non_trainable_params)} ({non_trainable_params:,})")
    
    # Calculate model size
    param_size_mb = (total_params * 4) / (1024 * 1024)  # Assuming float32
    console.print(f"[bold]Model Size:[/bold] ~{param_size_mb:.1f} MB\n")


def confirm_action(message: str, default: bool = False) -> bool:
    """Ask for user confirmation"""
    console = Console()
    
    if default:
        prompt = f"{message} [Y/n]: "
    else:
        prompt = f"{message} [y/N]: "
    
    response = console.input(prompt).lower()
    
    if not response:
        return default
    
    return response in ['y', 'yes']


def display_training_config(config: Dict[str, Any]):
    """Display training configuration in a nice format"""
    console = Console()
    
    # Group configuration by category
    categories = {
        'Model': ['model_type', 'num_classes', 'input_size', 'pretrained_weights'],
        'Training': ['epochs', 'batch_size', 'base_lr', 'weight_decay'],
        'Optimization': ['optimizer', 'use_sam', 'use_ema', 'use_amp'],
        'Augmentation': ['use_mixup', 'use_cutmix', 'use_tta'],
        'Paths': ['dataset_path', 'output_path', 'checkpoint_path']
    }
    
    for category, keys in categories.items():
        table = Table(title=category, show_header=False, box=None)
        table.add_column("Parameter", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        
        for key in keys:
            if key in config:
                value = config[key]
                if isinstance(value, Path):
                    value = str(value)
                elif isinstance(value, bool):
                    value = "✓" if value else "✗"
                table.add_row(key, str(value))
        
        console.print(table)
        console.print()


def create_checkpoint_table(checkpoint_dir: Path) -> Optional[Table]:
    """Create a table of available checkpoints"""
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = list(checkpoint_dir.glob("*.pth"))
    if not checkpoints:
        return None
    
    table = Table(title="Available Checkpoints", show_header=True)
    table.add_column("Filename", style="cyan")
    table.add_column("Size", style="green", justify="right")
    table.add_column("Modified", style="yellow")
    
    for ckpt in sorted(checkpoints, key=lambda x: x.stat().st_mtime, reverse=True):
        size_mb = ckpt.stat().st_size / (1024 * 1024)
        modified = ckpt.stat().st_mtime
        
        from datetime import datetime
        mod_time = datetime.fromtimestamp(modified).strftime("%Y-%m-%d %H:%M")
        
        table.add_row(
            ckpt.name,
            f"{size_mb:.1f} MB",
            mod_time
        )
    
    return table


def print_metrics_summary(metrics: Dict[str, float], title: str = "Metrics Summary"):
    """Print metrics in a formatted table"""
    console = Console()
    
    # Group metrics
    basic_metrics = ['accuracy', 'balanced_accuracy', 'cohen_kappa', 'auc_macro']
    medical_metrics = ['mean_sensitivity', 'mean_specificity', 'dr_quadratic_kappa']
    
    table = Table(title=title, show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    
    # Add basic metrics
    for metric in basic_metrics:
        if metric in metrics:
            value = metrics[metric]
            if metric in ['accuracy', 'balanced_accuracy', 'mean_sensitivity', 'mean_specificity']:
                formatted = f"{value:.2f}%"
            else:
                formatted = f"{value:.4f}"
            table.add_row(metric.replace('_', ' ').title(), formatted)
    
    # Add medical metrics if present
    for metric in medical_metrics:
        if metric in metrics:
            value = metrics[metric]
            if 'sensitivity' in metric or 'specificity' in metric:
                formatted = f"{value:.1f}%"
            else:
                formatted = f"{value:.3f}"
            table.add_row(metric.replace('_', ' ').title(), formatted)
    
    console.print(table)


def display_class_performance(metrics: Dict[str, float], class_names: List[str], 
                             top_n: int = 10):
    """Display per-class performance metrics"""
    console = Console()
    
    # Collect per-class metrics
    class_metrics = {}
    
    for class_name in class_names:
        class_data = {}
        for metric in ['sensitivity', 'specificity', 'f1', 'auc']:
            key = f"{class_name}_{metric}"
            if key in metrics:
                class_data[metric] = metrics[key]
        
        if class_data:
            class_metrics[class_name] = class_data
    
    if not class_metrics:
        return
    
    # Sort by sensitivity (critical for medical)
    sorted_classes = sorted(
        class_metrics.items(), 
        key=lambda x: x[1].get('sensitivity', 0),
        reverse=True
    )
    
    table = Table(title="Per-Class Performance", show_header=True)
    table.add_column("Class", style="cyan")
    table.add_column("Sensitivity", style="green", justify="right")
    table.add_column("Specificity", style="green", justify="right")
    table.add_column("F1-Score", style="yellow", justify="right")
    table.add_column("AUC", style="magenta", justify="right")
    
    for i, (class_name, metrics) in enumerate(sorted_classes[:top_n]):
        table.add_row(
            class_name,
            f"{metrics.get('sensitivity', 0):.1f}%" if 'sensitivity' in metrics else "-",
            f"{metrics.get('specificity', 0):.1f}%" if 'specificity' in metrics else "-",
            f"{metrics.get('f1', 0):.3f}" if 'f1' in metrics else "-",
            f"{metrics.get('auc', 0):.3f}" if 'auc' in metrics else "-"
        )
    
    console.print(table)
    
    if len(sorted_classes) > top_n:
        console.print(f"\n[dim]Showing top {top_n} of {len(sorted_classes)} classes[/dim]")


def create_error_report(exception: Exception, context: Dict[str, Any] = None):
    """Create detailed error report"""
    console = Console()
    
    import traceback
    
    # Error panel
    error_message = f"[bold red]Error:[/bold red] {type(exception).__name__}: {str(exception)}"
    console.print(Panel(error_message, title="Error Report", border_style="red"))
    
    # Traceback
    tb = traceback.format_exc()
    syntax = Syntax(tb, "python", theme="monokai", line_numbers=True)
    console.print("\n[bold]Traceback:[/bold]")
    console.print(syntax)
    
    # Context information
    if context:
        console.print("\n[bold]Context:[/bold]")
        for key, value in context.items():
            console.print(f"  {key}: {value}")
    
    # Suggestions
    console.print("\n[bold]Suggestions:[/bold]")
    
    if "CUDA out of memory" in str(exception):
        console.print("  • Reduce batch size")
        console.print("  • Enable gradient checkpointing")
        console.print("  • Use mixed precision training")
        console.print("  • Close other GPU applications")
    
    elif "No such file or directory" in str(exception):
        console.print("  • Check file paths are correct")
        console.print("  • Ensure dataset is properly downloaded")
        console.print("  • Verify permissions on directories")
    
    elif "RuntimeError" in str(exception) and "shape" in str(exception):
        console.print("  • Check input image dimensions")
        console.print("  • Verify model configuration matches checkpoint")
        console.print("  • Ensure preprocessing is correct")


def setup_file_logging(log_file: Path, level: str = "INFO"):
    """Setup file logging with rotation"""
    from logging.handlers import RotatingFileHandler
    
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    
    file_handler.setLevel(getattr(logging, level.upper()))
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    
    logging.getLogger().addHandler(file_handler)


class ProgressTracker:
    """Track and display training progress"""
    
    def __init__(self, total_epochs: int, total_steps: int):
        self.console = Console()
        self.total_epochs = total_epochs
        self.total_steps = total_steps
        self.current_epoch = 0
        self.current_step = 0
        self.start_time = None
        
    def start_epoch(self, epoch: int):
        """Start tracking new epoch"""
        self.current_epoch = epoch
        self.current_step = 0
        self.start_time = time.time()
        
    def update_step(self, metrics: Dict[str, float]):
        """Update step progress"""
        self.current_step += 1
        
        # Calculate ETA
        if self.start_time:
            elapsed = time.time() - self.start_time
            steps_per_sec = self.current_step / elapsed
            remaining_steps = self.total_steps - self.current_step
            eta = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
            
            # Update display
            self.console.print(
                f"Epoch [{self.current_epoch}/{self.total_epochs}] "
                f"Step [{self.current_step}/{self.total_steps}] "
                f"Loss: {metrics.get('loss', 0):.4f} "
                f"LR: {metrics.get('lr', 0):.2e} "
                f"ETA: {format_time(eta)}",
                end='\r'
            )
    
    def end_epoch(self, metrics: Dict[str, float]):
        """End epoch and display summary"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        self.console.print()  # New line
        self.console.print(
            f"[green]✓[/green] Epoch {self.current_epoch} completed in {format_time(elapsed)} - "
            f"Train Loss: {metrics.get('train_loss', 0):.4f}, "
            f"Val Loss: {metrics.get('val_loss', 0):.4f}, "
            f"Val Acc: {metrics.get('val_acc', 0):.2f}%"
        )


# Click decorators for common options
def common_options(func):
    """Add common CLI options to commands"""
    func = click.option('--verbose', '-v', is_flag=True, help='Verbose output')(func)
    func = click.option('--quiet', '-q', is_flag=True, help='Quiet mode')(func)
    func = click.option('--log-file', type=click.Path(), help='Log file path')(func)
    func = click.option('--no-color', is_flag=True, help='Disable colored output')(func)
    return func


def gpu_options(func):
    """Add GPU-related options to commands"""
    func = click.option('--device', '-d', default='cuda', help='Device to use (cuda/cpu)')(func)
    func = click.option('--gpu-id', type=int, default=0, help='GPU ID to use')(func)
    func = click.option('--no-cuda', is_flag=True, help='Disable CUDA even if available')(func)
    return func


def path_options(func):
    """Add path-related options to commands"""
    func = click.option('--data-path', type=click.Path(exists=True), help='Dataset path')(func)
    func = click.option('--output-path', type=click.Path(), help='Output directory')(func)
    func = click.option('--cache-dir', type=click.Path(), help='Cache directory')(func)
    return func
