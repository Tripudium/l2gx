"""
ORBITAAL Bitcoin Temporal Graph Dataset Demonstration.

This example demonstrates the ORBITAAL dataset for financial anomaly detection
and temporal graph analysis on Bitcoin transactions.

Features demonstrated:
- Temporal Bitcoin transaction graph analysis
- Financial anomaly detection capabilities
- Entity-based transaction modeling
- Time-series analysis of cryptocurrency flows
- Fraud detection patterns in Bitcoin networks
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
from l2gx.datasets import get_dataset


def demonstrate_orbitaal_loading():
    """Demonstrate loading and basic properties of ORBITAAL dataset."""
    print("₿ ORBITAAL BITCOIN TEMPORAL GRAPH DATASET")
    print("=" * 60)
    
    # Load ORBITAAL dataset
    print("🔄 Loading ORBITAAL dataset...")
    start_time = time.time()
    orbitaal = get_dataset("ORBITAAL", subset="sample")
    load_time = time.time() - start_time
    
    print(f"✅ Loaded in {load_time:.2f}s")
    print(f"📊 Dataset: {orbitaal}")
    print()
    
    # Get dataset statistics
    stats = orbitaal.get_statistics()
    print("📈 Dataset Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    print()
    
    return orbitaal


def analyze_temporal_patterns(orbitaal):
    """Analyze temporal patterns in Bitcoin transactions."""
    print("⏰ TEMPORAL TRANSACTION ANALYSIS")
    print("=" * 60)
    
    # Get transaction data
    edge_df, node_df = orbitaal.to("polars")
    
    print(f"📊 Transaction Data:")
    print(f"   Transactions: {len(edge_df):,}")
    print(f"   Entities: {len(node_df):,}")
    print(f"   Time span: {edge_df['timestamp'].min()} to {edge_df['timestamp'].max()}")
    print()
    
    # Analyze transaction amounts
    print("💰 Transaction Amount Analysis:")
    btc_amounts = edge_df['btc_amount']
    usd_amounts = edge_df['usd_amount']
    
    print(f"   BTC amounts - Mean: {btc_amounts.mean():.4f}, Median: {btc_amounts.median():.4f}")
    print(f"   USD amounts - Mean: ${usd_amounts.mean():,.2f}, Median: ${usd_amounts.median():,.2f}")
    print(f"   Largest BTC transaction: {btc_amounts.max():.4f} BTC")
    print(f"   Largest USD transaction: ${usd_amounts.max():,.2f}")
    print()
    
    # Time-based analysis
    print("📅 Temporal Distribution:")
    
    # Convert timestamps to readable dates
    timestamps = edge_df['timestamp'].to_numpy()
    dates = [datetime.fromtimestamp(ts) for ts in timestamps[:10]]  # Sample first 10
    
    print(f"   Sample transaction times:")
    for i, date in enumerate(dates):
        btc_amt = edge_df['btc_amount'][i]
        usd_amt = edge_df['usd_amount'][i]
        print(f"     {date.strftime('%Y-%m-%d %H:%M')}: {btc_amt:.4f} BTC (${usd_amt:,.2f})")
    print()
    
    return edge_df, node_df


def analyze_entity_types(orbitaal):
    """Analyze different entity types in the Bitcoin network."""
    print("🏢 ENTITY TYPE ANALYSIS")
    print("=" * 60)
    
    _, node_df = orbitaal.to("polars")
    
    # Get entity type distribution
    entity_types = ['exchange', 'wallet', 'service', 'miner', 'unknown']
    
    print("🏷️  Entity Type Distribution:")
    for entity_type in entity_types:
        count = node_df.filter(node_df['entity_type'] == entity_type).height
        percentage = count / len(node_df) * 100
        print(f"   {entity_type.capitalize():>8}: {count:4d} entities ({percentage:5.1f}%)")
    print()
    
    # Analyze labeled vs unlabeled entities
    labeled_count = node_df.filter(node_df['is_labeled'] == True).height
    unlabeled_count = len(node_df) - labeled_count
    
    print("🏷️  Entity Labeling:")
    print(f"   Labeled entities:   {labeled_count:4d} ({labeled_count/len(node_df)*100:5.1f}%)")
    print(f"   Unlabeled entities: {unlabeled_count:4d} ({unlabeled_count/len(node_df)*100:5.1f}%)")
    print()


def analyze_anomaly_detection(orbitaal):
    """Demonstrate anomaly detection capabilities."""
    print("🚨 FINANCIAL ANOMALY DETECTION")
    print("=" * 60)
    
    # Get anomaly labels
    anomaly_labels = orbitaal.get_anomaly_labels()
    edge_df, _ = orbitaal.to("polars")
    
    if anomaly_labels is not None:
        total_transactions = len(anomaly_labels)
        anomalies = anomaly_labels.filter(anomaly_labels['is_anomaly'] == True)
        normal_transactions = total_transactions - len(anomalies)
        
        print("📊 Anomaly Statistics:")
        print(f"   Total transactions: {total_transactions:,}")
        print(f"   Normal transactions: {normal_transactions:,} ({normal_transactions/total_transactions*100:.1f}%)")
        print(f"   Anomalous transactions: {len(anomalies):,} ({len(anomalies)/total_transactions*100:.1f}%)")
        print()
        
        if len(anomalies) > 0:
            print("🔍 Anomaly Examples:")
            
            # Get details of anomalous transactions
            anomaly_ids = anomalies['transaction_id'].to_list()[:5]  # First 5 anomalies
            
            for i, anomaly_id in enumerate(anomaly_ids):
                transaction = edge_df.filter(edge_df['transaction_id'] == anomaly_id).row(0)
                src, dst, timestamp, btc_amount, usd_amount, is_anomaly, tx_id = transaction
                
                date = datetime.fromtimestamp(timestamp)
                print(f"   Anomaly {i+1}:")
                print(f"     Transaction ID: {tx_id}")
                print(f"     Date: {date.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"     From Entity: {src} → To Entity: {dst}")
                print(f"     Amount: {btc_amount:.4f} BTC (${usd_amount:,.2f})")
                print()
        
        # Compare normal vs anomalous transaction amounts
        normal_transactions_df = edge_df.filter(edge_df['is_anomaly'] == 0)
        anomalous_transactions_df = edge_df.filter(edge_df['is_anomaly'] == 1)
        
        if len(anomalous_transactions_df) > 0:
            print("💱 Amount Comparison (Normal vs Anomalous):")
            print(f"   Normal transactions:")
            print(f"     Mean BTC: {normal_transactions_df['btc_amount'].mean():.4f}")
            print(f"     Mean USD: ${normal_transactions_df['usd_amount'].mean():,.2f}")
            print(f"   Anomalous transactions:")
            print(f"     Mean BTC: {anomalous_transactions_df['btc_amount'].mean():.4f}")
            print(f"     Mean USD: ${anomalous_transactions_df['usd_amount'].mean():,.2f}")
            
            ratio = (anomalous_transactions_df['usd_amount'].mean() / 
                    normal_transactions_df['usd_amount'].mean())
            print(f"   Anomalous transactions are {ratio:.1f}x larger on average")
            print()
    
    else:
        print("❌ No anomaly labels available")


def analyze_network_structure(orbitaal):
    """Analyze the network structure and connectivity."""
    print("🌐 NETWORK STRUCTURE ANALYSIS")
    print("=" * 60)
    
    # Convert to different graph formats
    raphtory_graph = orbitaal.to("raphtory")
    edge_df, node_df = orbitaal.to("polars")
    
    print("📊 Graph Statistics:")
    print(f"   Nodes: {raphtory_graph.count_nodes():,}")
    print(f"   Edges: {raphtory_graph.count_edges():,}")
    print(f"   Temporal edges: {raphtory_graph.count_temporal_edges():,}")
    print()
    
    # Analyze degree distribution
    print("📈 Network Connectivity:")
    
    # Calculate in-degree and out-degree from edge list
    in_degrees = edge_df.group_by('dst').len().sort('len', descending=True)
    out_degrees = edge_df.group_by('src').len().sort('len', descending=True)
    
    print(f"   Most active recipients (in-degree):")
    for i in range(min(5, len(in_degrees))):
        entity_id = in_degrees[i, 'dst']
        in_degree = in_degrees[i, 'len']
        print(f"     Entity {entity_id}: {in_degree} incoming transactions")
    
    print(f"   Most active senders (out-degree):")
    for i in range(min(5, len(out_degrees))):
        entity_id = out_degrees[i, 'src']
        out_degree = out_degrees[i, 'len']
        print(f"     Entity {entity_id}: {out_degree} outgoing transactions")
    print()


def demonstrate_temporal_graph_features(orbitaal):
    """Demonstrate temporal graph-specific features."""
    print("⏱️  TEMPORAL GRAPH FEATURES")
    print("=" * 60)
    
    raphtory_graph = orbitaal.to("raphtory")
    
    print("🕐 Temporal Properties:")
    print(f"   Earliest transaction: {raphtory_graph.earliest_date_time}")
    print(f"   Latest transaction: {raphtory_graph.latest_date_time}")
    
    # Analyze temporal density
    edge_df, _ = orbitaal.to("polars")
    time_span = edge_df['timestamp'].max() - edge_df['timestamp'].min()
    avg_transactions_per_second = len(edge_df) / time_span
    
    print(f"   Time span: {time_span:,} seconds")
    print(f"   Average transactions per second: {avg_transactions_per_second:.4f}")
    print()
    
    print("📊 Temporal Analysis Capabilities:")
    print("   ✅ Transaction time series analysis")
    print("   ✅ Entity behavior over time")
    print("   ✅ Anomaly detection in temporal context")
    print("   ✅ Dynamic network evolution")
    print("   ✅ Time-window based fraud detection")
    print()


def run_orbitaal_demo():
    """Run the complete ORBITAAL dataset demonstration."""
    print("₿ ORBITAAL BITCOIN TEMPORAL GRAPH DEMONSTRATION")
    print("=" * 80)
    print()
    
    try:
        # 1. Load dataset
        orbitaal = demonstrate_orbitaal_loading()
        
        # 2. Temporal analysis
        edge_df, node_df = analyze_temporal_patterns(orbitaal)
        
        # 3. Entity analysis
        analyze_entity_types(orbitaal)
        
        # 4. Anomaly detection
        analyze_anomaly_detection(orbitaal)
        
        # 5. Network structure
        analyze_network_structure(orbitaal)
        
        # 6. Temporal features
        demonstrate_temporal_graph_features(orbitaal)
        
        # Summary
        print("🎊 ORBITAAL DEMONSTRATION SUMMARY")
        print("=" * 60)
        print("✅ Successfully demonstrated:")
        print("   • Bitcoin temporal transaction graph loading")
        print("   • Financial anomaly detection capabilities")
        print("   • Entity-based transaction modeling")
        print("   • Temporal pattern analysis")
        print("   • Network structure analysis")
        print("   • Fraud detection in cryptocurrency networks")
        print()
        print("📊 Dataset Applications:")
        print("   • Financial fraud detection")
        print("   • Temporal graph neural networks")
        print("   • Cryptocurrency flow analysis")
        print("   • Anomaly detection research")
        print("   • Anti-money laundering (AML)")
        print()
        print("🚀 ORBITAAL is ready for financial anomaly detection research!")
        
        return orbitaal
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = run_orbitaal_demo()