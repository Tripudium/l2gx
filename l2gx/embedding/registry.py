"""
Registry system for graph embedding methods.

This module provides a centralized registry for discovering and instantiating
different graph embedding methods with a consistent interface.
"""

from typing import Dict, Type, List
from .base import GraphEmbedding


class EmbeddingRegistry:
    """
    Registry for graph embedding methods.
    
    Provides a centralized way to register, discover, and instantiate
    graph embedding methods.
    """
    
    def __init__(self):
        self._methods: Dict[str, Type[GraphEmbedding]] = {}
        self._aliases: Dict[str, str] = {}
        
    def register(self, 
                 name: str, 
                 embedding_class: Type[GraphEmbedding],
                 aliases: List[str] = None) -> None:
        """
        Register a graph embedding method.
        
        Args:
            name: Primary name for the embedding method
            embedding_class: Class implementing GraphEmbedding interface
            aliases: Alternative names for the method
        """
        if not issubclass(embedding_class, GraphEmbedding):
            raise TypeError(f"{embedding_class} must inherit from GraphEmbedding")
            
        self._methods[name] = embedding_class
        
        if aliases:
            for alias in aliases:
                self._aliases[alias] = name
                
    def get(self, name: str) -> Type[GraphEmbedding]:
        """
        Get an embedding method class by name or alias.
        
        Args:
            name: Name or alias of the embedding method
            
        Returns:
            Embedding method class
            
        Raises:
            KeyError: If method name/alias is not registered
        """
        # Check aliases first
        if name in self._aliases:
            name = self._aliases[name]
            
        if name not in self._methods:
            available = list(self._methods.keys()) + list(self._aliases.keys())
            raise KeyError(f"Unknown embedding method '{name}'. Available: {available}")
            
        return self._methods[name]
    
    def create(self, name: str, **kwargs) -> GraphEmbedding:
        """
        Create an instance of an embedding method.
        
        Args:
            name: Name or alias of the embedding method
            **kwargs: Parameters to pass to the embedding constructor
            
        Returns:
            Initialized embedding method instance
        """
        embedding_class = self.get(name)
        return embedding_class(**kwargs)
    
    def list_methods(self) -> List[str]:
        """
        List all registered embedding method names.
        
        Returns:
            List of method names
        """
        return list(self._methods.keys())
    
    def list_aliases(self) -> Dict[str, str]:
        """
        List all aliases and their corresponding method names.
        
        Returns:
            Dictionary mapping aliases to method names
        """
        return self._aliases.copy()
    
    def remove(self, name: str) -> None:
        """
        Remove a method from the registry.
        
        Args:
            name: Name of the method to remove
        """
        if name in self._methods:
            del self._methods[name]
            
        # Remove any aliases pointing to this method
        aliases_to_remove = [alias for alias, method in self._aliases.items() 
                           if method == name]
        for alias in aliases_to_remove:
            del self._aliases[alias]


# Global registry instance
EMBEDDING_REGISTRY = EmbeddingRegistry()


def register_embedding(name: str, aliases: List[str] = None):
    """
    Decorator to register embedding methods.
    
    Args:
        name: Primary name for the embedding method
        aliases: Alternative names for the method
        
    Example:
        @register_embedding('my_embedding', aliases=['my_emb'])
        class MyEmbedding(GraphEmbedding):
            pass
    """
    def decorator(embedding_class: Type[GraphEmbedding]):
        EMBEDDING_REGISTRY.register(name, embedding_class, aliases)
        return embedding_class
    return decorator


def get_embedding(name: str, **kwargs) -> GraphEmbedding:
    """
    Convenience function to create embedding instances.
    
    Args:
        name: Name or alias of the embedding method
        **kwargs: Parameters to pass to the embedding constructor
        
    Returns:
        Initialized embedding method instance
    """
    return EMBEDDING_REGISTRY.create(name, **kwargs)


def list_embeddings() -> List[str]:
    """
    List all available embedding methods.
    
    Returns:
        List of available embedding method names
    """
    return EMBEDDING_REGISTRY.list_methods()