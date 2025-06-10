"""Registry pattern implementation for extensible components."""

from typing import Dict, Type, Any, Optional, Callable, TypeVar, Generic
import logging

from retfound.core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

T = TypeVar('T')


class Registry(Generic[T]):
    """Generic registry for registering and retrieving components.
    
    This allows for extensible architecture where new models, optimizers,
    losses, etc. can be easily added without modifying core code.
    """
    
    def __init__(self, name: str):
        """Initialize registry.
        
        Args:
            name: Name of the registry (e.g., 'models', 'optimizers')
        """
        self.name = name
        self._registry: Dict[str, Type[T]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
    
    def register(
        self,
        name: str,
        cls: Type[T] = None,
        **metadata: Any
    ) -> Callable:
        """Register a class with optional metadata.
        
        Can be used as a decorator or called directly.
        
        Args:
            name: Name to register the class under
            cls: Class to register (optional if used as decorator)
            **metadata: Additional metadata to store with the class
            
        Returns:
            Decorator function if cls is None, otherwise the class
        """
        def decorator(cls_to_register: Type[T]) -> Type[T]:
            if name in self._registry:
                logger.warning(
                    f"Overwriting existing registration '{name}' in {self.name} registry"
                )
            
            self._registry[name] = cls_to_register
            self._metadata[name] = metadata
            
            logger.debug(f"Registered '{name}' in {self.name} registry")
            return cls_to_register
        
        if cls is None:
            return decorator
        else:
            return decorator(cls)
    
    def get(self, name: str) -> Type[T]:
        """Get a registered class by name.
        
        Args:
            name: Name of the registered class
            
        Returns:
            The registered class
            
        Raises:
            ConfigurationError: If name is not registered
        """
        if name not in self._registry:
            available = list(self._registry.keys())
            raise ConfigurationError(
                f"'{name}' not found in {self.name} registry. "
                f"Available options: {available}"
            )
        
        return self._registry[name]
    
    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for a registered class.
        
        Args:
            name: Name of the registered class
            
        Returns:
            Metadata dictionary
        """
        return self._metadata.get(name, {})
    
    def create(self, name: str, **kwargs: Any) -> T:
        """Create an instance of a registered class.
        
        Args:
            name: Name of the registered class
            **kwargs: Arguments to pass to the class constructor
            
        Returns:
            Instance of the registered class
        """
        cls = self.get(name)
        return cls(**kwargs)
    
    def list(self) -> list[str]:
        """List all registered names.
        
        Returns:
            List of registered names
        """
        return list(self._registry.keys())
    
    def items(self) -> list[tuple[str, Type[T]]]:
        """Get all registered items.
        
        Returns:
            List of (name, class) tuples
        """
        return list(self._registry.items())
    
    def __contains__(self, name: str) -> bool:
        """Check if a name is registered.
        
        Args:
            name: Name to check
            
        Returns:
            True if registered, False otherwise
        """
        return name in self._registry
    
    def __len__(self) -> int:
        """Get number of registered items.
        
        Returns:
            Number of registered items
        """
        return len(self._registry)
    
    def __repr__(self) -> str:
        """String representation of the registry.
        
        Returns:
            String representation
        """
        return f"Registry(name='{self.name}', items={self.list()})"


# Global registries
_registries: Dict[str, Registry] = {}


def get_registry(name: str) -> Registry:
    """Get or create a registry by name.
    
    Args:
        name: Name of the registry
        
    Returns:
        Registry instance
    """
    if name not in _registries:
        _registries[name] = Registry(name)
    return _registries[name]


def register(registry_name: str, name: str, **metadata: Any) -> Callable:
    """Convenience decorator for registering to a specific registry.
    
    Args:
        registry_name: Name of the registry to register to
        name: Name to register under
        **metadata: Additional metadata
        
    Returns:
        Decorator function
    """
    registry = get_registry(registry_name)
    return registry.register(name, **metadata)


# Pre-create common registries
MODEL_REGISTRY = get_registry("models")
OPTIMIZER_REGISTRY = get_registry("optimizers")
LOSS_REGISTRY = get_registry("losses")
SCHEDULER_REGISTRY = get_registry("schedulers")
TRANSFORM_REGISTRY = get_registry("transforms")
CALLBACK_REGISTRY = get_registry("callbacks")
METRIC_REGISTRY = get_registry("metrics")
EXPORT_REGISTRY = get_registry("exporters")


# Example usage in docstring
"""
Example usage:

    # Register a model
    @register("models", "my_custom_model", paper="https://arxiv.org/...")
    class MyCustomModel(nn.Module):
        def __init__(self, num_classes: int):
            super().__init__()
            # ...
    
    # Register an optimizer
    @OPTIMIZER_REGISTRY.register("my_optimizer")
    class MyOptimizer(torch.optim.Optimizer):
        def __init__(self, params, lr=1e-3):
            # ...
    
    # Use registered components
    model_cls = MODEL_REGISTRY.get("my_custom_model")
    model = model_cls(num_classes=10)
    
    # List available models
    available_models = MODEL_REGISTRY.list()
"""