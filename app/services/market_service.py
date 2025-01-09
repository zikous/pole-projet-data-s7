# services/market_service.py
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime
import logging
from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def convert_to_serializable(obj):
    """Convert numpy/pandas objects to JSON-serializable types"""
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, pd.DataFrame):
        return {str(k): convert_to_serializable(v) for k, v in obj.to_dict().items()}
    elif isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, np.ndarray)):
        return [convert_to_serializable(x) for x in obj]
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj


def get_correlation_and_clusters(tickers, start_date, end_date):
    """
    Fetch stock data, calculate correlation matrix, and perform clustering
    """
    try:
        # Validate and clean tickers
        tickers = [ticker.strip().upper() for ticker in tickers]

        # Download data for all tickers
        data = pd.DataFrame()
        returns_data = pd.DataFrame()

        failed_tickers = []
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date)
                if hist.empty:
                    failed_tickers.append(ticker)
                    continue
                data[ticker] = hist["Close"]
                returns_data[ticker] = hist["Close"].pct_change().fillna(0)
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {str(e)}")
                failed_tickers.append(ticker)

        if len(failed_tickers) == len(tickers):
            return {
                "success": False,
                "error": "Could not fetch data for any of the provided tickers",
            }

        if failed_tickers:
            logger.warning(f"Failed to fetch data for tickers: {failed_tickers}")

        # Remove failed tickers from the list
        tickers = [t for t in tickers if t not in failed_tickers]

        # Calculate correlation matrix
        corr_matrix = data.corr()

        # Perform clustering
        clustering_results = perform_clustering(returns_data)

        # Create visualizations
        corr_plot = create_correlation_plot(corr_matrix, clustering_results)
        cluster_plot = create_cluster_plot(returns_data, clustering_results)

        # Convert all data to JSON-serializable format
        result = {
            "success": True,
            "correlation_matrix": convert_to_serializable(corr_matrix),
            "correlation_plot": corr_plot.to_json(),
            "cluster_plot": cluster_plot.to_json(),
            "clusters": convert_to_serializable(clustering_results),
            "failed_tickers": failed_tickers,
        }

        return result

    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}


def perform_clustering(returns_data):
    """
    Perform Affinity Propagation clustering on stock returns
    """
    # Prepare data for clustering
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(returns_data)

    # Apply Affinity Propagation
    af = AffinityPropagation(random_state=42, damping=0.9)
    cluster_labels = af.fit_predict(scaled_returns.T)

    # Prepare clustering results with serializable types
    clustering_results = {
        "labels": [int(x) for x in cluster_labels],
        "n_clusters": int(len(set(cluster_labels))),
        "clusters": {},
        "exemplars": [],
    }

    # Group stocks by cluster
    for i, ticker in enumerate(returns_data.columns):
        cluster_num = int(cluster_labels[i])
        if cluster_num not in clustering_results["clusters"]:
            clustering_results["clusters"][cluster_num] = []
        clustering_results["clusters"][cluster_num].append(str(ticker))

        # Identify exemplars (cluster centers)
        if i in af.cluster_centers_indices_:
            clustering_results["exemplars"].append(str(ticker))

    return clustering_results


# The rest of the visualization functions remain the same


def create_correlation_plot(corr_matrix, clustering_results):
    """
    Create correlation heatmap with cluster information
    """
    # Sort tickers by cluster
    sorted_tickers = []
    for cluster in sorted(clustering_results["clusters"].keys()):
        sorted_tickers.extend(clustering_results["clusters"][cluster])

    # Reorder correlation matrix
    corr_matrix = corr_matrix.loc[sorted_tickers, sorted_tickers]

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.index,
            y=corr_matrix.columns,
            hoverongaps=False,
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            text=np.round(corr_matrix.values, decimals=2),
            texttemplate="%{text}",
            textfont={"size": 10},
            showscale=True,
        )
    )

    # Add cluster separators
    current_pos = 0
    for cluster in sorted(clustering_results["clusters"].keys()):
        cluster_size = len(clustering_results["clusters"][cluster])
        if current_pos > 0:
            # Add vertical line
            fig.add_shape(
                type="line",
                x0=current_pos - 0.5,
                x1=current_pos - 0.5,
                y0=-0.5,
                y1=len(corr_matrix) - 0.5,
                line=dict(color="white", width=2),
            )
            # Add horizontal line
            fig.add_shape(
                type="line",
                x0=-0.5,
                x1=len(corr_matrix) - 0.5,
                y0=current_pos - 0.5,
                y1=current_pos - 0.5,
                line=dict(color="white", width=2),
            )
        current_pos += cluster_size

    fig.update_layout(
        title="Stock Correlation Matrix (Clustered)",
        xaxis_title="Stocks",
        yaxis_title="Stocks",
        width=800,
        height=800,
        template="plotly_dark",
    )

    return fig


def create_cluster_plot(returns_data, clustering_results):
    """
    Create a scatter plot of stocks based on their first two principal components
    """
    from sklearn.decomposition import PCA

    # Prepare data for visualization
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(returns_data)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(
        scaled_returns.T
    )  # Transpose to transform stocks, not timestamps

    # Create DataFrame for plotting
    plot_df = pd.DataFrame(
        pca_results, columns=["PC1", "PC2"], index=returns_data.columns
    )
    plot_df["Cluster"] = clustering_results["labels"]
    plot_df["Exemplar"] = plot_df.index.isin(clustering_results["exemplars"])

    # Create scatter plot
    fig = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        color="Cluster",
        text=plot_df.index,
        title="Stock Clusters based on Return Patterns",
        template="plotly_dark",
    )

    # Highlight exemplars
    exemplar_df = plot_df[plot_df["Exemplar"]]
    fig.add_trace(
        go.Scatter(
            x=exemplar_df["PC1"],
            y=exemplar_df["PC2"],
            mode="markers",
            marker=dict(symbol="star", size=15, line=dict(color="white", width=2)),
            name="Cluster Centers",
            showlegend=True,
        )
    )

    fig.update_traces(textposition="top center", marker=dict(size=10))

    fig.update_layout(width=800, height=600, showlegend=True)

    return fig
