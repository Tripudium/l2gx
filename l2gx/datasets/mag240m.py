"""
MAG240M Dataset.

The MAG240M dataset is a large heterogeneous academic citation graph extracted 
from the Microsoft Academic Graph (MAG). It contains over 240 million nodes 
representing papers, authors, institutions, and fields of study.

Reference: https://ogb.stanford.edu/docs/lsc/mag240m/
Paper: Hu et al. "Open Graph Benchmark: Datasets for Machine Learning on Graphs" (2020)
"""

from pathlib import Path
from typing import Optional, Callable

try:
    from ogb.lsc import MAG240MDataset as OGBDataset
except ImportError:
    OGBDataset = None

from .registry import register_dataset
from .base import BaseDataset


@register_dataset("MAG240M")
class MAG240MDataset(BaseDataset):
    """
    MAG240M heterogeneous academic citation dataset.
    
    This is a large-scale dataset containing:
    - 244+ million nodes (papers, authors, institutions, fields of study)
    - 1.7+ billion edges (citations, authorship, affiliation, etc.)
    - Node features and publication year information
    
    Note: This dataset is very large (~100GB) and requires the OGB library.
    """
    
    def __init__(
        self,
        root: str | None = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        **kwargs
    ):
        """
        Initialize the MAG240M dataset.
        
        Args:
            root: Root directory for dataset storage
            transform: Optional data transformation function
            pre_transform: Optional preprocessing transformation function
            **kwargs: Additional arguments passed to OGB dataset
            
        Note:
            Requires the ogb library: pip install ogb
        """
        if OGBDataset is None:
            raise ImportError(
                "MAG240M dataset requires the OGB library. "
                "Install with: pip install ogb"
            )
        
        # Logger is set up in BaseDataset
        
        if root is None:
            root = str(Path(__file__).parent.parent.parent / "data" / "mag240m")
        
        super().__init__(root, transform, pre_transform)
        
        # Initialize OGB dataset
        self.logger.info("Loading MAG240M dataset (this may take a while...)")
        self.ogb_dataset = OGBDataset(root=root, **kwargs)
        
        # Note: This dataset is too large to fully load into memory
        # We provide methods to access subsets and statistics
        self.logger.info("MAG240M dataset initialized")

    @property
    def raw_file_names(self) -> list[str]:
        """Raw files are managed by OGB."""
        return []

    @property
    def processed_file_names(self) -> list[str]:
        """Processed files for subsets."""
        return ["subset_edge_data.parquet", "subset_node_data.parquet"]

    def download(self):
        """Download is handled by OGB automatically."""
        pass

    def process(self):
        """
        Process a subset of the MAG240M dataset for demonstration.
        
        Since the full dataset is too large, we extract a manageable subset
        centered around recent papers in computer science.
        """
        processed_dir = Path(self.processed_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Skip if already processed
        if all((processed_dir / fname).exists() for fname in self.processed_file_names):
            self.logger.info("Subset already processed")
            return
        
        self.logger.info("Processing MAG240M subset...")
        
        # For now, we'll create a placeholder implementation
        # In practice, this would extract a subset of papers and their citations
        self.logger.warning(
            "MAG240M subset processing not fully implemented. "
            "This dataset requires specialized handling due to its size."
        )
        
        # Create empty placeholder files
        import polars as pl
        empty_edges = pl.DataFrame({
            'src': [], 'dst': [], 'timestamp': []
        })
        empty_nodes = pl.DataFrame({
            'id': [], 'timestamp': []
        })
        
        empty_edges.write_parquet(processed_dir / "subset_edge_data.parquet")
        empty_nodes.write_parquet(processed_dir / "subset_node_data.parquet")
        
        self.logger.info("Placeholder subset created")

    def get_paper_features(self, paper_indices):
        """
        Get features for specific papers.
        
        Args:
            paper_indices: List of paper indices
            
        Returns:
            Feature matrix for the specified papers
        """
        return self.ogb_dataset.paper_feat[paper_indices]

    def get_citation_graph(self):
        """
        Get the citation graph edge index.
        
        Returns:
            Edge index tensor for citations
        """
        return self.ogb_dataset.edge_index('paper', 'cites', 'paper')

    def get_author_paper_edges(self):
        """
        Get author-paper relationship edges.
        
        Returns:
            Edge index tensor for author-paper relationships
        """
        return self.ogb_dataset.edge_index('author', 'writes', 'paper')

    def get_num_papers(self):
        """Get the number of papers in the dataset."""
        return self.ogb_dataset.num_papers

    def get_num_authors(self):
        """Get the number of authors in the dataset."""
        return self.ogb_dataset.num_authors

    def get_statistics(self):
        """
        Get dataset statistics.
        
        Returns:
            Dictionary containing dataset statistics
        """
        if not hasattr(self, '_stats'):
            self._stats = {
                'num_papers': self.get_num_papers(),
                'num_authors': self.get_num_authors(),
                'num_institutions': getattr(self.ogb_dataset, 'num_institutions', 0),
                'num_fields': getattr(self.ogb_dataset, 'num_fields', 0),
            }
        return self._stats

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"{self.__class__.__name__}("
            f"papers={stats['num_papers']}, "
            f"authors={stats['num_authors']})"
        )