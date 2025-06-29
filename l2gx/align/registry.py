"""
Registry for aligners.
"""

from typing import Type

ALIGNER_REGISTRY: dict[str, Type] = {}


def register_aligner(name):
    """
    Decorator to register an aligner class under a specified name.

    Parameters:
        name (str): The key under which the aligner class will be registered.

    Returns:
        A decorator that registers the class.
    """

    def decorator(cls):
        ALIGNER_REGISTRY[name] = cls
        return cls

    return decorator


def get_aligner(name, **kwargs):
    """
    Factory function to instantiate an aligner based on its registered name.

    Parameters:
        name (str): The registered name of the aligner (e.g., "local2global").
        **kwargs: Additional keyword arguments that will be passed to the aligner's constructor.

    Returns:
        An instance of the aligner class corresponding to the given name.

    Raises:
        ValueError: If the aligner name is not found in the registry.
    """
    if name not in ALIGNER_REGISTRY:
        available = list(ALIGNER_REGISTRY.keys())
        raise ValueError(
            f"Aligner '{name}' is not registered. Available aligners: {available}"
        )

    return ALIGNER_REGISTRY[name](**kwargs)
