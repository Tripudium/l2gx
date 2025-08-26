"""
Utilities for converting between different graph formats.
"""

import numpy as np
import polars as pl
from typing import Optional, Callable
import torch
from torch import Tensor
from torch_geometric.data import Data, TemporalData
from torch_geometric.utils import coalesce
from raphtory import Graph  # pylint: disable=no-name-in-module
import networkx as nx


def polars_to_tg(
    edge_df: pl.DataFrame,
    node_df: pl.DataFrame = None,
    pre_transform: Optional[Callable] = None,
) -> tuple[Data, Optional[dict[str, Tensor]]]:
    """
    Convert a pair of Polars DataFrames (edge and node) to a list of PyTorch Geometric Data objects.
    """
    # Process nodes first to get the number of nodes and node features
    num_nodes = max(int(edge_df["src"].max() + 1), int(edge_df["dst"].max() + 1))
    if node_df is not None:
        feature_cols = [
            col for col in node_df.columns if col not in ["timestamp", "id", "y"]
        ]
        if feature_cols:
            # Group by node id, take first occurrence of features, and sort by id
            node_features = (
                node_df.group_by("id")
                .agg([pl.col(col).first() for col in feature_cols])
                .sort("id")
                .select(feature_cols)
                .to_numpy()
            )
            x = torch.from_numpy(node_features)
        else:
            x = None
        # Process labels
        if "y" in node_df.columns:
            y = node_df["y"].to_numpy()
        else:
            y = None
    else:
        x = None
    # Process edges for each timestamp
    data_list = []
    unique_timestamps = sorted(edge_df["timestamp"].unique().to_list())
    for ts in unique_timestamps:
        # Process edges
        edges_filtered = (
            edge_df.filter(pl.col("timestamp") == ts).select(["src", "dst"]).sort("src")
        )  # Sort by source column for adjacency index efficiency
        edge_array = edges_filtered.to_numpy()
        if edge_array.size == 0:
            continue
        edge_index = torch.from_numpy(edge_array).t()
        # Process edge features
        edge_feature_cols = [
            col
            for col in edges_filtered.columns
            if col not in ["timestamp", "src", "dst"]
        ]
        if edge_feature_cols:
            edge_features = edges_filtered.select(edge_feature_cols).to_numpy()
            edge_attr = torch.from_numpy(edge_features)
        else:
            edge_attr = None
        if edge_attr is not None:
            edge_index = coalesce(edge_index, edge_attr, num_nodes=num_nodes)
        else:
            edge_index = coalesce(edge_index, num_nodes=num_nodes)
        data = Data(edge_index=edge_index, num_nodes=num_nodes, timestamp=ts)
        if x is not None:
            data.x = x
        if edge_attr is not None:
            data.edge_attr = edge_attr
        if y is not None:
            data.y = torch.from_numpy(y)
        data.timestamp = torch.tensor(ts)
        data_list.append(data)
    if pre_transform is not None:
        data_list = [pre_transform(data) for data in data_list]
    return data_list


def polars_to_raphtory(edge_df: pl.DataFrame, node_df: pl.DataFrame = None) -> Graph:
    """
    Convert a Polars DataFrame to a Raphtory graph.
    """
    graph = Graph()
    # Convert Polars DataFrames to Pandas DataFrames
    edge_df_pd = edge_df.to_pandas()
    if node_df is not None:
        node_df_pd = node_df.to_pandas()
    else:
        node_df_pd = None

    graph.load_edges_from_pandas(
        edge_df_pd,
        time="timestamp",
        src="src",
        dst="dst",
        properties=[
            col for col in edge_df.columns if col not in ["timestamp", "src", "dst"]
        ],
    )
    if node_df is not None:
        graph.load_nodes_from_pandas(
            df=node_df_pd,
            time="timestamp",
            id="id",
            constant_properties=[
                c for c in node_df.columns if c not in ["timestamp", "id"]
            ],
        )
    return graph


def tg_to_polars(data_list: list[Data]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Convert a list of PyTorch Geometric Data objects to a Polars DataFrame.
    """
    edge_dfs = []
    node_dfs = []
    for data in data_list:
        # Handle edges
        edge_index = data.edge_index.t().numpy()
        edge_dict = {"src": edge_index[:, 0], "dst": edge_index[:, 1]}
        if hasattr(data, "timestamp"):
            edge_dict["timestamp"] = data.timestamp
        else:
            edge_dict["timestamp"] = np.int64(0)
        if hasattr(data, "edge_attr"):
            if data.edge_attr is not None:
                edge_attr = data.edge_attr.numpy()
                for i in range(edge_attr.shape[1]):
                    edge_dict[f"edge_feature_{i}"] = edge_attr[:, i]
        edge_dfs.append(pl.DataFrame(edge_dict))

        # Handle nodes
        node_dict = None
        if hasattr(data, "x") and data.x is not None:
            node_features = data.x.numpy()
            node_dict = {"id": range(data.num_nodes)}
            if hasattr(data, "timestamp"):
                node_dict["timestamp"] = data.timestamp
            else:
                node_dict["timestamp"] = np.int64(0)  # type: ignore
            for i in range(node_features.shape[1]):
                node_dict[f"node_feature_{i}"] = node_features[:, i]

            # Handle labels
            if hasattr(data, "y"):
                if data.y is not None:
                    node_dict["y"] = data.y.numpy()

        if node_dict:
            node_dfs.append(pl.DataFrame(node_dict))

    edge_df = pl.concat(edge_dfs)
    node_df = pl.concat(node_dfs)

    return edge_df, node_df


def polars_to_networkx(edge_df: pl.DataFrame, node_df: pl.DataFrame = None) -> nx.Graph:
    """
    Convert Polars DataFrames to a NetworkX graph.

    Args:
        edge_df: DataFrame with columns 'src', 'dst', and optional 'weight' and other edge attributes
        node_df: Optional DataFrame with node features and attributes

    Returns:
        NetworkX Graph object
    """
    # Determine if graph should be directed based on edge structure
    # For simplicity, assume undirected unless specified otherwise
    G = nx.Graph()

    # Add nodes
    if node_df is not None:
        node_data = node_df.to_pandas()
        for _, row in node_data.iterrows():
            node_id = row.get("id", row.name)
            # Add node attributes (excluding 'id' and 'timestamp')
            attrs = {k: v for k, v in row.items() if k not in ["id", "timestamp"]}
            G.add_node(node_id, **attrs)

    # Add edges
    edge_data = edge_df.to_pandas()
    for _, row in edge_data.iterrows():
        src = row["src"]
        dst = row["dst"]

        # Add edge attributes (excluding 'src', 'dst', 'timestamp')
        attrs = {k: v for k, v in row.items() if k not in ["src", "dst", "timestamp"]}

        # Handle weight specially if present
        if "weight" in attrs:
            G.add_edge(
                src,
                dst,
                weight=attrs["weight"],
                **{k: v for k, v in attrs.items() if k != "weight"},
            )
        else:
            G.add_edge(src, dst, **attrs)

    return G


def polars_to_temporal_data(
    edge_df: pl.DataFrame, node_df: pl.DataFrame = None
) -> TemporalData:
    """
    Convert Polars DataFrames to PyTorch Geometric TemporalData format.

    This function creates a continuous-time dynamic graph representation
    where each edge represents a temporal event with a timestamp.

    Args:
        edge_df: DataFrame with required columns 'src', 'dst', 'timestamp'
                 and optional edge features
        node_df: Optional DataFrame with node features (currently not used
                 in TemporalData format)

    Returns:
        TemporalData object representing the temporal graph as event stream
    """
    # Sort by timestamp to maintain temporal order
    edge_df = edge_df.sort("timestamp")

    # Extract required temporal data attributes
    src = torch.tensor(edge_df["src"].to_numpy(), dtype=torch.long)
    dst = torch.tensor(edge_df["dst"].to_numpy(), dtype=torch.long)
    t = torch.tensor(edge_df["timestamp"].to_numpy(), dtype=torch.long)

    # Create TemporalData object
    temporal_data = TemporalData(src=src, dst=dst, t=t)

    # Add optional edge features as 'msg' if present
    feature_columns = [
        col for col in edge_df.columns if col not in ["src", "dst", "timestamp"]
    ]

    if feature_columns:
        # Stack all feature columns into a single tensor
        features = []
        for col in feature_columns:
            col_data = edge_df[col].to_numpy()
            # Convert to appropriate tensor type
            if edge_df[col].dtype in [pl.Float32, pl.Float64]:
                features.append(torch.tensor(col_data, dtype=torch.float))
            elif edge_df[col].dtype in [pl.Int32, pl.Int64]:
                features.append(torch.tensor(col_data, dtype=torch.long))
            elif edge_df[col].dtype == pl.Boolean:
                features.append(torch.tensor(col_data, dtype=torch.bool))
            else:
                # Default to float for other types
                features.append(torch.tensor(col_data.astype(float), dtype=torch.float))

        # Stack features into message tensor
        if len(features) == 1:
            temporal_data.msg = features[0].unsqueeze(1)
        else:
            # Stack multiple features
            temporal_data.msg = torch.stack(features, dim=1)

    return temporal_data
