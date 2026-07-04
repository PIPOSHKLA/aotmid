"""Social Network Analysis page."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from aot_stock_network.network_analysis import NetworkBuilder, NetworkAnalyzer, NetworkVisualizer
from aot_stock_network.dashboard.utils import get_data, inject_css, download_figure, download_button

st.set_page_config(page_title="Social Network Analysis", page_icon="🕸️", layout="wide")
inject_css()

st.title("🕸️ Social Network Analysis")
st.markdown("Factor interdependence graph with centrality and community detection.")

df = get_data()
if df is None or df.empty:
    st.warning("No data available.")
    st.stop()

method = st.selectbox("Correlation method", ["pearson", "spearman", "mutual_info"], index=0)
threshold = st.slider("Threshold", 0.0, 1.0, 0.3, 0.05)

builder = NetworkBuilder(df)
G = builder.build_graph(method=method, threshold=threshold)
analyzer = NetworkAnalyzer(G)
metrics = analyzer.compute_all()
viz = NetworkVisualizer(G, metrics)

col1, col2 = st.columns([2, 1])

with col1:
    fig, ax = viz.plot_static(figsize=(10, 7))
    st.pyplot(fig)
    svg = viz.export_svg(fig) if hasattr(viz, "export_svg") else None
    download_figure(fig, "network_graph.png")

with col2:
    st.subheader("Network Metrics")
    st.metric("Nodes", G.number_of_nodes())
    st.metric("Edges", G.number_of_edges())
    st.metric("Density", f"{metrics.network_density:.3f}")
    st.metric("Communities", metrics.n_communities)
    st.metric("Connected Components", metrics.connected_components)

    if metrics.average_path_length:
        st.metric("Avg Path Length", f"{metrics.average_path_length:.3f}")
    if metrics.diameter:
        st.metric("Diameter", f"{metrics.diameter:.3f}")

st.subheader("Centrality Metrics")
df_cent = metrics.to_dataframe()
st.dataframe(df_cent.style.highlight_max(), use_container_width=True)

st.subheader("Community Membership")
comm_df = pd.DataFrame(list(metrics.community.items()), columns=["Node", "Community"])
st.dataframe(comm_df, use_container_width=True)
