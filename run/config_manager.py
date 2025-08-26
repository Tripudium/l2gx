"""
Configuration Management System for L2GX Experiments.

This module provides utilities for loading, validating, and managing
experiment configurations stored in YAML files.
"""

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import any

import yaml


@dataclass
class DatasetConfig:
    """Dataset configuration parameters."""

    name: str = "Cora"
    use_default_splits: bool = True
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    split_seed: int = 42


@dataclass
class EmbeddingConfig:
    """Embedding configuration parameters."""

    method: str = "vgae"
    embedding_dim: int = 64
    hidden_dim: int = 32
    num_epochs: int = 200
    learning_rate: float = 0.01
    dropout: float = 0.5
    graphsage_aggregator: str = "mean"
    dgi_encoder: str = "gcn"
    svd_matrix_type: str = "normalized"


@dataclass
class PatchConfig:
    """Patch configuration parameters."""

    num_patches: int = 10
    clustering_method: str = "metis"
    min_overlap: int = 27
    target_overlap: int = 54
    sparsification_method: str = "resistance"
    sparsification_params: dict[str, any] = field(
        default_factory=lambda: {
            "resistance_alpha": 0.1,
            "sampling_rate": 0.8,
            "k_neighbors": 5,
        }
    )


@dataclass
class AlignmentConfig:
    """Alignment configuration parameters."""

    method: str = "geo2"
    num_epochs: int = 1000
    learning_rate: float = 0.01
    use_orthogonal_reg: bool = True
    orthogonal_reg_weight: float = 100.0
    use_bfs_training: bool = True
    center_patches: bool = False
    min_overlap: int = 27


@dataclass
class NodeReconstructionConfig:
    """Node reconstruction task configuration."""

    loss_function: str = "mse"
    evaluation_metrics: list[str] = field(
        default_factory=lambda: ["mse", "mae", "cosine_similarity"]
    )
    reconstruction_method: str = "autoencoder"
    decoder_hidden_dims: list[int] = field(default_factory=lambda: [32, 64])


@dataclass
class NodeClassificationConfig:
    """Node classification task configuration."""

    classifier: str = "logistic_regression"
    mlp_hidden_dims: list[int] = field(default_factory=lambda: [32])
    mlp_num_epochs: int = 100
    mlp_learning_rate: float = 0.001
    mlp_dropout: float = 0.2
    svm_kernel: str = "rbf"
    svm_C: float = 1.0
    rf_n_estimators: int = 100
    rf_max_depth: int = 10
    evaluation_metrics: list[str] = field(
        default_factory=lambda: [
            "accuracy",
            "f1_macro",
            "f1_micro",
            "precision",
            "recall",
        ]
    )
    stratify: bool = True


@dataclass
class HyperparameterSearchConfig:
    """Hyperparameter search configuration."""

    enabled: bool = False
    method: str = "grid"
    n_trials: int = 50
    search_space: dict[str, list[any]] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """General experiment configuration."""

    name: str = "cora_embedding_experiment"
    random_seed: int = 42
    num_runs: int = 5
    device: str = "auto"
    verbose: bool = True
    save_intermediate: bool = True
    output_dir: str = "experiments/results"


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    log_to_file: bool = True
    log_file: str = "experiments/logs/experiment.log"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class VisualizationConfig:
    """Visualization configuration."""

    enabled: bool = True
    methods: list[str] = field(default_factory=lambda: ["tsne", "umap"])
    save_plots: bool = True
    plot_format: str = "png"
    plot_dpi: int = 300


@dataclass
class Config:
    """Main configuration class that holds all configuration sections."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    patches: PatchConfig = field(default_factory=PatchConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    node_reconstruction: NodeReconstructionConfig = field(
        default_factory=NodeReconstructionConfig
    )
    node_classification: NodeClassificationConfig = field(
        default_factory=NodeClassificationConfig
    )
    hyperparameter_search: HyperparameterSearchConfig = field(
        default_factory=HyperparameterSearchConfig
    )
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


class ConfigManager:
    """Configuration manager for loading and validating experiment configurations."""

    VALID_EMBEDDING_METHODS = ["gae", "vgae", "svd", "graphsage", "dgi"]
    VALID_CLUSTERING_METHODS = ["metis", "louvain", "fennel", "hierarchical"]
    VALID_ALIGNMENT_METHODS = ["geo", "geo2", "l2g"]
    VALID_DEVICES = ["cpu", "cuda", "auto"]

    def __init__(self, config_path: str | None = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config: Config | None = None

    def load_config(self, config_path: str | None = None) -> Config:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Loaded and validated configuration
        """
        if config_path is None:
            config_path = self.config_path

        if config_path is None:
            raise ValueError("No configuration path provided")

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        # Convert dictionary to Config object
        self.config = self._dict_to_config(config_dict)

        # Validate configuration
        self._validate_config(self.config)

        return self.config

    def save_config(self, config: Config, save_path: str):
        """
        Save configuration to YAML file.

        Args:
            config: Configuration object to save
            save_path: Path where to save the configuration
        """
        config_dict = self._config_to_dict(config)

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    def _dict_to_config(self, config_dict: dict[str, any]) -> Config:
        """Convert dictionary to Config object."""
        config = Config()

        # Update each section if present in the dictionary
        if "dataset" in config_dict:
            config.dataset = DatasetConfig(**config_dict["dataset"])

        if "embedding" in config_dict:
            config.embedding = EmbeddingConfig(**config_dict["embedding"])

        if "patches" in config_dict:
            config.patches = PatchConfig(**config_dict["patches"])

        if "alignment" in config_dict:
            config.alignment = AlignmentConfig(**config_dict["alignment"])

        if "node_reconstruction" in config_dict:
            config.node_reconstruction = NodeReconstructionConfig(
                **config_dict["node_reconstruction"]
            )

        if "node_classification" in config_dict:
            config.node_classification = NodeClassificationConfig(
                **config_dict["node_classification"]
            )

        if "hyperparameter_search" in config_dict:
            config.hyperparameter_search = HyperparameterSearchConfig(
                **config_dict["hyperparameter_search"]
            )

        if "experiment" in config_dict:
            config.experiment = ExperimentConfig(**config_dict["experiment"])

        if "logging" in config_dict:
            config.logging = LoggingConfig(**config_dict["logging"])

        if "visualization" in config_dict:
            config.visualization = VisualizationConfig(**config_dict["visualization"])

        return config

    def _config_to_dict(self, config: Config) -> dict[str, any]:
        """Convert Config object to dictionary."""
        return {
            "dataset": config.dataset.__dict__,
            "embedding": config.embedding.__dict__,
            "patches": config.patches.__dict__,
            "alignment": config.alignment.__dict__,
            "node_reconstruction": config.node_reconstruction.__dict__,
            "node_classification": config.node_classification.__dict__,
            "hyperparameter_search": config.hyperparameter_search.__dict__,
            "experiment": config.experiment.__dict__,
            "logging": config.logging.__dict__,
            "visualization": config.visualization.__dict__,
        }

    def _validate_config(self, config: Config):
        """Validate configuration parameters."""
        # Validate embedding method
        if config.embedding.method not in self.VALID_EMBEDDING_METHODS:
            raise ValueError(
                f"Invalid embedding method: {config.embedding.method}. "
                f"Valid options: {self.VALID_EMBEDDING_METHODS}"
            )

        # Validate clustering method
        if config.patches.clustering_method not in self.VALID_CLUSTERING_METHODS:
            raise ValueError(
                f"Invalid clustering method: {config.patches.clustering_method}. "
                f"Valid options: {self.VALID_CLUSTERING_METHODS}"
            )

        # Validate alignment method
        if config.alignment.method not in self.VALID_ALIGNMENT_METHODS:
            raise ValueError(
                f"Invalid alignment method: {config.alignment.method}. "
                f"Valid options: {self.VALID_ALIGNMENT_METHODS}"
            )

        # Validate device
        if config.experiment.device not in self.VALID_DEVICES:
            raise ValueError(
                f"Invalid device: {config.experiment.device}. "
                f"Valid options: {self.VALID_DEVICES}"
            )

        # Validate split ratios
        total_ratio = (
            config.dataset.train_ratio
            + config.dataset.val_ratio
            + config.dataset.test_ratio
        )
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(
                f"Train/val/test ratios must sum to 1.0, got {total_ratio}"
            )

        # Validate dimensions are positive
        if config.embedding.embedding_dim <= 0:
            raise ValueError("Embedding dimension must be positive")

        if config.embedding.hidden_dim <= 0:
            raise ValueError("Hidden dimension must be positive")

        # Validate patch parameters
        if config.patches.num_patches <= 0:
            raise ValueError("Number of patches must be positive")

        if config.patches.min_overlap < 0:
            raise ValueError("Minimum overlap must be non-negative")

        if config.patches.target_overlap < config.patches.min_overlap:
            raise ValueError("Target overlap must be >= minimum overlap")

    def get_embedding_params(self) -> dict[str, any]:
        """Get parameters for embedding method initialization."""
        if self.config is None:
            raise ValueError("Configuration not loaded")

        base_params = {
            "embedding_dim": self.config.embedding.embedding_dim,
        }

        # Add method-specific parameters
        method = self.config.embedding.method
        if method in ["gae", "vgae", "graphsage", "dgi"]:
            base_params.update(
                {
                    "hidden_dim": self.config.embedding.hidden_dim,
                    "epochs": self.config.embedding.num_epochs,
                    "learning_rate": self.config.embedding.learning_rate,
                    "dropout": self.config.embedding.dropout,
                }
            )

        if method == "graphsage":
            base_params["aggregator"] = self.config.embedding.graphsage_aggregator

        if method == "dgi":
            base_params["encoder_type"] = self.config.embedding.dgi_encoder

        if method == "svd":
            base_params["matrix_type"] = self.config.embedding.svd_matrix_type

        return base_params

    def get_patch_params(self) -> dict[str, any]:
        """Get parameters for patch generation."""
        if self.config is None:
            raise ValueError("Configuration not loaded")

        return {
            "num_patches": self.config.patches.num_patches,
            "clustering_method": self.config.patches.clustering_method,
            "min_overlap": self.config.patches.min_overlap,
            "target_overlap": self.config.patches.target_overlap,
            "sparsification_method": self.config.patches.sparsification_method,
            **self.config.patches.sparsification_params,
        }

    def get_alignment_params(self) -> dict[str, any]:
        """Get parameters for alignment method."""
        if self.config is None:
            raise ValueError("Configuration not loaded")

        return {
            "num_epochs": self.config.alignment.num_epochs,
            "learning_rate": self.config.alignment.learning_rate,
            "use_orthogonal_reg": self.config.alignment.use_orthogonal_reg,
            "orthogonal_reg_weight": self.config.alignment.orthogonal_reg_weight,
            "use_bfs_training": self.config.alignment.use_bfs_training,
            "center_patches": self.config.alignment.center_patches,
            "min_overlap": self.config.alignment.min_overlap,
        }

    def create_output_directories(self):
        """Create necessary output directories."""
        if self.config is None:
            raise ValueError("Configuration not loaded")

        # Create output directory
        output_dir = Path(self.config.experiment.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (output_dir / "embeddings").mkdir(exist_ok=True)
        (output_dir / "patches").mkdir(exist_ok=True)
        (output_dir / "alignments").mkdir(exist_ok=True)
        (output_dir / "results").mkdir(exist_ok=True)
        (output_dir / "plots").mkdir(exist_ok=True)

        # Create log directory
        if self.config.logging.log_to_file:
            log_path = Path(self.config.logging.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

    def setup_logging(self):
        """Setup logging based on configuration."""
        if self.config is None:
            raise ValueError("Configuration not loaded")

        log_config = self.config.logging

        # Configure logging
        log_level = getattr(logging, log_config.level.upper())

        handlers = []

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        handlers.append(console_handler)

        # File handler
        if log_config.log_to_file:
            file_handler = logging.FileHandler(log_config.log_file)
            file_handler.setLevel(log_level)
            handlers.append(file_handler)

        # Configure logging
        logging.basicConfig(
            level=log_level, format=log_config.format, handlers=handlers, force=True
        )

    def get_experiment_variants(self) -> list[Config]:
        """
        Generate configuration variants for hyperparameter search.

        Returns:
            list of configuration variants
        """
        if self.config is None:
            raise ValueError("Configuration not loaded")

        if not self.config.hyperparameter_search.enabled:
            return [self.config]

        search_config = self.config.hyperparameter_search
        search_space = search_config.search_space

        if search_config.method == "grid":
            return self._generate_grid_search_configs(search_space)
        elif search_config.method == "random":
            return self._generate_random_search_configs(
                search_space, search_config.n_trials
            )
        else:
            # For Optuna, return base config (Optuna will handle parameter generation)
            return [self.config]

    def _generate_grid_search_configs(
        self, search_space: dict[str, list[any]]
    ) -> list[Config]:
        """Generate configurations for grid search."""
        import itertools

        if not search_space:
            return [self.config]

        # Get parameter names and values
        param_names = list(search_space.keys())
        param_values = list(search_space.values())

        # Generate all combinations
        configs = []
        for combination in itertools.product(*param_values):
            config_copy = copy.deepcopy(self.config)

            # Apply parameter values
            for param_name, param_value in zip(param_names, combination, strict=False):
                self._set_config_parameter(config_copy, param_name, param_value)

            configs.append(config_copy)

        return configs

    def _generate_random_search_configs(
        self, search_space: dict[str, list[any]], n_trials: int
    ) -> list[Config]:
        """Generate configurations for random search."""
        import random

        if not search_space:
            return [self.config]

        configs = []
        for _ in range(n_trials):
            config_copy = copy.deepcopy(self.config)

            # Randomly sample parameters
            for param_name, param_values in search_space.items():
                param_value = random.choice(param_values)
                self._set_config_parameter(config_copy, param_name, param_value)

            configs.append(config_copy)

        return configs

    def _set_config_parameter(self, config: Config, param_name: str, param_value: any):
        """set a parameter value in the configuration."""
        # Map parameter names to config attributes
        param_mapping = {
            "embedding_dim": ("embedding", "embedding_dim"),
            "hidden_dim": ("embedding", "hidden_dim"),
            "learning_rate": ("embedding", "learning_rate"),
            "num_epochs": ("embedding", "num_epochs"),
            "num_patches": ("patches", "num_patches"),
        }

        if param_name in param_mapping:
            section, attr = param_mapping[param_name]
            setattr(getattr(config, section), attr, param_value)
        else:
            raise ValueError(f"Unknown parameter: {param_name}")


def load_config(config_path: str) -> Config:
    """
    Convenience function to load configuration.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Loaded configuration
    """
    manager = ConfigManager()
    return manager.load_config(config_path)


def create_default_config() -> Config:
    """
    Create a default configuration.

    Returns:
        Default configuration
    """
    return Config()


if __name__ == "__main__":
    # Example usage
    config_manager = ConfigManager()

    # Create default config and save it
    default_config = create_default_config()
    config_manager.save_config(default_config, "experiments/default_config.yaml")
    print("Created default configuration at: experiments/default_config.yaml")

    # Load and validate existing config
    try:
        config = config_manager.load_config("experiments/config.yaml")
        print("✅ Configuration loaded and validated successfully")

        # Print some key parameters
        print(f"Embedding method: {config.embedding.method}")
        print(f"Embedding dimension: {config.embedding.embedding_dim}")
        print(f"Number of patches: {config.patches.num_patches}")
        print(f"Output directory: {config.experiment.output_dir}")

    except Exception as e:
        print(f"❌ Configuration error: {e}")
