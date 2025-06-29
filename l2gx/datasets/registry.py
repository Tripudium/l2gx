"""
Registry for datasets.
"""

DATASET_REGISTRY = {}


def register_dataset(name: str):
    """
    Decorator to register a dataset class under a specified name.

    Parameters:
        name (str): The key under which the dataset class will be registered.

    Returns:
        A decorator that registers the class.
    """

    def decorator(cls):
        DATASET_REGISTRY[name] = cls
        return cls

    return decorator


def get_dataset(name: str, **kwargs):
    """
    Factory function to instantiate a dataset based on its registered name.

    Parameters:
        name (str): The registered name of the dataset (e.g., "Cora", "as-733").
        **kwargs: Additional keyword arguments that will be passed to the dataset's constructor.

    Returns:
        An instance of the dataset class corresponding to the given name.

    Raises:
        ValueError: If the dataset name is not found in the registry.
        ImportError: If required dependencies are missing for the dataset.
        FileNotFoundError: If required data files are not found.

    Examples:
        >>> # Load Cora dataset
        >>> cora = get_dataset("Cora")
        
        >>> # Load AS-733 with custom root directory
        >>> as733 = get_dataset("as-733", root="/path/to/data")
        
        >>> # Load Elliptic with source file
        >>> elliptic = get_dataset("Elliptic", source_file="/path/to/elliptic.zip")
    """
    if name not in DATASET_REGISTRY:
        available = sorted(DATASET_REGISTRY.keys())
        raise ValueError(
            f"Dataset '{name}' is not registered. "
            f"Available datasets: {available}. "
            f"Use list_available_datasets() to see all options."
        )

    try:
        return DATASET_REGISTRY[name](**kwargs)
    except ImportError as e:
        raise ImportError(
            f"Failed to load dataset '{name}' due to missing dependencies. "
            f"Original error: {e}"
        ) from e
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Failed to load dataset '{name}' due to missing files. "
            f"Original error: {e}. "
            f"Check that all required data files are available."
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize dataset '{name}': {e}"
        ) from e
