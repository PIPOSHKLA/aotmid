"""Tests for :mod:`aot_stock_network.network_analysis`."""

from __future__ import annotations

import pytest

from aot_stock_network.network_analysis import (
    NetworkAnalyzer,
    NetworkBuilder,
    NetworkMetrics,
    NetworkVisualizer,
    run_network_pipeline,
)


class TestNetworkBuilder:
    """NetworkBuilder constructs correlation graphs from DataFrames."""

    def test_pearson_build(self, sample_df: "pd.DataFrame") -> None:
        builder = NetworkBuilder(df=sample_df)
        G = builder.build_graph(method="pearson", threshold=0.3)
        assert G.number_of_nodes() > 0

    def test_spearman_build(self, sample_df: "pd.DataFrame") -> None:
        builder = NetworkBuilder(df=sample_df)
        G = builder.build_graph(method="spearman", threshold=0.3)
        assert G.number_of_nodes() > 0

    def test_mutual_info_build(self, sample_df: "pd.DataFrame") -> None:
        builder = NetworkBuilder(df=sample_df)
        G = builder.build_graph(method="mutual_info", threshold=0.3)
        assert G.number_of_nodes() > 0

    def test_adjacency_matrix(self, sample_df: "pd.DataFrame") -> None:
        builder = NetworkBuilder(df=sample_df)
        builder.build_graph(method="pearson", threshold=0.3)
        adj = builder.adjacency_matrix()
        assert adj.shape[0] == adj.shape[1]

    def test_correlation_matrix(self, sample_df: "pd.DataFrame") -> None:
        builder = NetworkBuilder(df=sample_df)
        builder.build_graph(method="pearson", threshold=0.3)
        corr = builder.correlation_matrix()
        assert corr.shape[0] == corr.shape[1]

    def test_summary(self, sample_df: "pd.DataFrame") -> None:
        builder = NetworkBuilder(df=sample_df)
        builder.build_graph()
        s = builder.summary()
        assert "Nodes" in s


class TestNetworkAnalyzer:
    """NetworkAnalyzer computes graph metrics."""

    @pytest.fixture(autouse=True)
    def _setup(self, sample_df: "pd.DataFrame") -> None:
        G = NetworkBuilder(df=sample_df).build_graph()
        self.analyzer = NetworkAnalyzer(G)

    def test_centrality_types(self) -> None:
        m = self.analyzer.compute_all()
        assert isinstance(m.degree_centrality, dict)
        assert len(m.degree_centrality) > 0

    def test_community_detection(self) -> None:
        m = self.analyzer.compute_all()
        assert m.n_communities >= 1
        assert len(m.community) > 0

    def test_summary(self) -> None:
        self.analyzer.compute_all()
        s = self.analyzer.summary()
        assert "NetworkAnalysis Summary" in s

    def test_metrics_property(self) -> None:
        self.analyzer.compute_all()
        assert isinstance(self.analyzer.metrics, NetworkMetrics)


class TestNetworkMetrics:
    """NetworkMetrics aggregates graph-level statistics."""

    @pytest.fixture(autouse=True)
    def _setup(self, sample_df: "pd.DataFrame") -> None:
        G = NetworkBuilder(df=sample_df).build_graph()
        analyzer = NetworkAnalyzer(G)
        self.metrics = analyzer.compute_all()

    def test_density(self) -> None:
        assert 0.0 <= self.metrics.network_density <= 1.0

    def test_connected_components(self) -> None:
        assert self.metrics.connected_components >= 1

    def test_to_dataframe(self) -> None:
        df = self.metrics.to_dataframe()
        assert "degree_centrality" in df.columns


class TestNetworkVisualizer:
    """Visualizer creates static/interactive plots."""

    @pytest.fixture(autouse=True)
    def _setup(self, sample_df: "pd.DataFrame") -> None:
        builder = NetworkBuilder(df=sample_df)
        G = builder.build_graph()
        analyzer = NetworkAnalyzer(G)
        metrics = analyzer.compute_all()
        self.viz = NetworkVisualizer(G, metrics)

    def test_static_plot(self) -> None:
        from matplotlib.figure import Figure

        fig, ax = self.viz.plot_static(figsize=(6, 4))
        assert isinstance(fig, Figure)


class TestRunNetworkPipeline:
    """End-to-end pipeline smoke test."""

    def test_standard_run(self, sample_df: "pd.DataFrame") -> None:
        result = run_network_pipeline(
            sample_df,
            method="pearson",
            threshold=0.3,
            generate_exports=False,
        )
        assert "builder" in result
        assert "metrics" in result
        assert "analyzer" in result
        assert "visualizer" in result
        assert result["metrics"].network_density > 0
