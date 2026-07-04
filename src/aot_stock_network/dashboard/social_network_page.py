"""Social Network Analysis: interactive graph, centrality metrics, community detection."""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st

from aot_stock_network.dashboard.utils import (
    download_button,
    download_figure,
    get_data,
    inject_css,
)
from aot_stock_network.network_analysis import (
    NetworkAnalyzer,
    NetworkBuilder,
    NetworkVisualizer,
)


@st.cache_resource(show_spinner="Building network graph...")
def build_network(df, method, threshold):
    """Build and analyze the network graph (cached)."""
    builder = NetworkBuilder(df)
    G = builder.build_graph(method=method, threshold=threshold, min_edges=1)
    analyzer = NetworkAnalyzer(G)
    metrics = analyzer.compute_all()
    viz = NetworkVisualizer(G, metrics)
    return builder, analyzer, metrics, viz


def show() -> None:
    inject_css(st.session_state.get("dark_mode", False))
    df = get_data()

    st.title("Social Network Analysis")
    st.markdown("Graph-theoretic analysis of factor interrelationships.")

    # ── Controls ──────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Network Controls")
        method = st.selectbox(
            "Edge weight method",
            ["pearson", "spearman", "mutual_info"],
            index=0,
            help="Association measure for edge weights.",
        )
        threshold = st.slider(
            "Edge threshold",
            0.0,
            1.0,
            0.3,
            0.05,
            help="Minimum |weight| to include an edge.",
        )

    # ── Build ─────────────────────────────────────────────
    if method and threshold is not None:
        builder, analyzer, metrics, viz = build_network(df, method, threshold)
        G = builder.graph

        # ── Summary metrics ────────────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Nodes", G.number_of_nodes())
        c2.metric("Edges", G.number_of_edges())
        c3.metric("Density", f"{nx.density(G):.3f}")
        c4.metric("Communities", metrics.n_communities)
        c5.metric("Components", metrics.connected_components)

        st.divider()

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "🌐 Network Graph",
                "📊 Centrality",
                "🏘️ Communities",
                "📈 Degree Distribution",
            ]
        )

        # ── Tab 1: Network Graph ──────────────────────────
        with tab1:
            st.markdown("#### Interactive Network Graph")
            st.markdown(
                "Hover for details, scroll to zoom, drag to pan. "
                "Node size ∝ degree, color = community."
            )
            html_path = Path(st.temporary_directory) / "network_interactive.html"
            try:
                viz.plot_interactive(html_path)
                with open(html_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                st.components.v1.html(html_content, height=600, scrolling=True)
            except Exception as exc:
                st.error(f"Interactive graph failed: {exc}")
                # Fallback: static
                fig, _ = viz.plot_static()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            st.markdown("#### Static Network Graph")
            fig, _ = viz.plot_static()
            st.pyplot(fig, use_container_width=True)
            download_figure(fig, "social_network_graph.svg")
            plt.close(fig)

        # ── Tab 2: Centrality ─────────────────────────────
        with tab2:
            c1, c2 = st.columns(2)
            with c1:
                fig_h, _ = viz.plot_centrality_heatmap()
                st.pyplot(fig_h, use_container_width=True)
                download_figure(fig_h, "centrality_heatmap.svg")
                plt.close(fig_h)
            with c2:
                fig_b, _ = viz.plot_centrality_barchart()
                st.pyplot(fig_b, use_container_width=True)
                download_figure(fig_b, "centrality_barchart.svg")
                plt.close(fig_b)

            with st.expander("📋 Centrality Metrics Table", expanded=False):
                df_metrics = metrics.to_dataframe()
                styled = df_metrics.style.background_gradient(
                    cmap="YlOrRd",
                    subset=["degree_centrality", "pagerank"],
                )
                st.dataframe(styled, use_container_width=True)
                download_button(
                    df_metrics.reset_index(),
                    "centrality_metrics.csv",
                )
                st.markdown(analyzer.summary())

        # ── Tab 3: Communities ────────────────────────────
        with tab3:
            st.markdown("#### Community Structure")
            st.markdown(
                f"Detected **{metrics.n_communities}** communities using the Louvain algorithm."
            )

            community_df = pd.DataFrame(
                {
                    "node": list(metrics.community.keys()),
                    "community": list(metrics.community.values()),
                }
            )
            for cid in sorted(community_df["community"].unique()):
                members = community_df[community_df["community"] == cid]["node"].tolist()
                st.markdown(f"**Community {cid}** ({len(members)} members): {', '.join(members)}")

            fig_adj, _ = viz.plot_adjacency_heatmap(builder.adjacency_matrix())
            st.pyplot(fig_adj, use_container_width=True)
            download_figure(fig_adj, "adjacency_heatmap.svg")
            plt.close(fig_adj)

        # ── Tab 4: Degree Distribution ────────────────────
        with tab4:
            fig_deg, _ = viz.plot_degree_distribution()
            st.pyplot(fig_deg, use_container_width=True)
            download_figure(fig_deg, "degree_distribution.svg")
            plt.close(fig_deg)

            # Node details
            st.markdown("#### Node Degree Details")
            deg_df = pd.DataFrame(
                {
                    "node": [n for n in G.nodes()],
                    "degree": [d for _, d in G.degree()],
                }
            ).sort_values("degree", ascending=False)
            st.dataframe(deg_df, use_container_width=True)

            if nx.is_connected(G):
                st.success("The graph is **fully connected** (1 component).")
            else:
                st.warning(
                    f"The graph has **{metrics.connected_components}** connected components."
                )

            if metrics.average_path_length is not None:
                st.metric("Average Path Length", f"{metrics.average_path_length:.4f}")
            if metrics.diameter is not None:
                st.metric("Graph Diameter", f"{metrics.diameter:.4f}")
