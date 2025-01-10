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

# Initialize the logger for this module
logger = logging.getLogger(__name__)  # Create a logger using the module's name


def convert_to_serializable(obj):
    """Convert numpy/pandas objects to JSON-serializable types"""

    # If the object is a numpy integer type (int64 or int32), convert it to a native Python integer
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)

    # If the object is a numpy float type (float64 or float32), convert it to a native Python float
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)

    # If the object is a pandas DataFrame, recursively convert its contents to serializable types
    elif isinstance(obj, pd.DataFrame):
        return {str(k): convert_to_serializable(v) for k, v in obj.to_dict().items()}

    # If the object is a dictionary, recursively convert each key-value pair to serializable types
    elif isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}

    # If the object is a list, tuple, or numpy ndarray, recursively convert each element to serializable types
    elif isinstance(obj, (list, tuple, np.ndarray)):
        return [convert_to_serializable(x) for x in obj]

    # If the object is a pandas Timestamp or a datetime object, convert it to an ISO format string
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()

    # For all other types, return the object as is (i.e., it is already serializable)
    return obj


def get_correlation_and_clusters(tickers, start_date, end_date):
    """
    Fetch stock data, calculate correlation matrix, and perform clustering
    """
    try:
        # Step 1: Validate and clean tickers by stripping spaces and converting to uppercase
        tickers = [ticker.strip().upper() for ticker in tickers]

        # Initialize empty data structures for storing stock price data and returns
        data = pd.DataFrame()
        returns_data = pd.DataFrame()

        # List to track tickers that failed to fetch data
        failed_tickers = []

        # Step 2: Download historical stock data for each ticker
        for ticker in tickers:
            try:
                # Fetch stock history for the given ticker using yfinance
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date)

                # If no data is available for the ticker, mark it as failed
                if hist.empty:
                    failed_tickers.append(ticker)
                    continue

                # Store the closing price and calculate the daily returns
                data[ticker] = hist["Close"]
                returns_data[ticker] = hist["Close"].pct_change().fillna(0)
            except Exception as e:
                # Log errors for tickers that failed to fetch data
                logger.error(f"Error fetching data for {ticker}: {str(e)}")
                failed_tickers.append(ticker)

        # Step 3: Check if all tickers failed to fetch data
        if len(failed_tickers) == len(tickers):
            return {
                "success": False,
                "error": "Could not fetch data for any of the provided tickers",
            }

        # Step 4: If some tickers failed, log a warning
        if failed_tickers:
            logger.warning(f"Failed to fetch data for tickers: {failed_tickers}")

        # Step 5: Remove failed tickers from the list for further processing
        tickers = [t for t in tickers if t not in failed_tickers]

        # Step 6: Calculate the correlation matrix for the fetched stock prices
        corr_matrix = data.corr()

        # Step 7: Perform clustering on the stock returns data
        clustering_results = perform_clustering(returns_data)

        # Step 8: Create visualizations for the correlation matrix and clusters
        corr_plot = create_correlation_plot(corr_matrix, clustering_results)
        cluster_plot = create_cluster_plot(returns_data, clustering_results)

        # Step 9: Convert all results to JSON-serializable format for the API response
        result = {
            "success": True,
            "correlation_matrix": convert_to_serializable(
                corr_matrix
            ),  # Serialize correlation matrix
            "correlation_plot": corr_plot.to_json(),  # Serialize correlation plot
            "cluster_plot": cluster_plot.to_json(),  # Serialize cluster plot
            "clusters": convert_to_serializable(
                clustering_results
            ),  # Serialize cluster results
            "failed_tickers": failed_tickers,  # Include tickers that failed
        }

        return result

    except Exception as e:
        # Log any unexpected errors and return an error response
        logger.error(f"Error in analysis: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}


def perform_clustering(returns_data):
    """
    Perform Affinity Propagation clustering on stock returns.
    The function groups stocks based on their return patterns and identifies exemplars (central representatives).
    """
    # Step 1: Prepare the data for clustering by scaling the returns data
    # StandardScaler normalizes the data to have mean 0 and variance 1, which helps in clustering
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(returns_data)

    # Step 2: Apply Affinity Propagation clustering algorithm
    # Affinity Propagation automatically finds the number of clusters based on a similarity matrix
    af = AffinityPropagation(
        random_state=42, damping=0.9
    )  # damping parameter helps with stability
    cluster_labels = af.fit_predict(
        scaled_returns.T
    )  # Transpose the returns data to cluster by stocks

    # Step 3: Prepare the clustering results in a JSON-serializable format
    clustering_results = {
        "labels": [
            int(x) for x in cluster_labels
        ],  # List of cluster labels for each stock
        "n_clusters": int(len(set(cluster_labels))),  # Number of unique clusters
        "clusters": {},  # Dictionary to store stocks in each cluster
        "exemplars": [],  # List to store exemplars (central representatives)
    }

    # Step 4: Group stocks by their cluster label
    for i, ticker in enumerate(returns_data.columns):
        cluster_num = int(cluster_labels[i])
        if cluster_num not in clustering_results["clusters"]:
            clustering_results["clusters"][cluster_num] = []
        clustering_results["clusters"][cluster_num].append(
            str(ticker)
        )  # Add stock ticker to the cluster

        # Step 5: Identify exemplars (cluster centers)
        # An exemplar is a stock that is the representative of the cluster
        if i in af.cluster_centers_indices_:
            clustering_results["exemplars"].append(
                str(ticker)
            )  # Add the exemplar to the list

    # Step 6: Return the clustering results
    return clustering_results


def create_correlation_plot(corr_matrix, clustering_results):
    """
    Create a correlation heatmap with cluster information. The heatmap shows how the stocks are correlated,
    with the clusters of stocks visually separated using lines.
    """
    # Step 1: Sort tickers based on their cluster labels
    # This ensures that stocks within the same cluster are placed together on the heatmap
    sorted_tickers = []
    for cluster in sorted(clustering_results["clusters"].keys()):
        sorted_tickers.extend(clustering_results["clusters"][cluster])

    # Step 2: Reorder the correlation matrix to match the sorted tickers
    # This ensures that the correlation values in the heatmap correspond to the sorted tickers
    corr_matrix = corr_matrix.loc[sorted_tickers, sorted_tickers]

    # Step 3: Create the heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,  # Correlation values as the heatmap's data
            x=corr_matrix.index,  # Stock tickers for the x-axis
            y=corr_matrix.columns,  # Stock tickers for the y-axis
            hoverongaps=False,  # Disable hover on empty cells
            colorscale="RdBu",  # Color scale representing the correlation (red/blue for negative/positive)
            zmin=-1,  # Minimum correlation value (for color scale)
            zmax=1,  # Maximum correlation value (for color scale)
            text=np.round(
                corr_matrix.values, decimals=2
            ),  # Display the correlation values as text in each cell
            texttemplate="%{text}",  # Format for displaying the text in each cell
            textfont={"size": 10},  # Set font size for the text in the cells
            showscale=True,  # Show the color scale on the side
        )
    )

    # Step 4: Add cluster separators as vertical and horizontal lines
    # These lines visually separate the clusters on the heatmap
    current_pos = 0
    for cluster in sorted(clustering_results["clusters"].keys()):
        cluster_size = len(clustering_results["clusters"][cluster])
        if current_pos > 0:
            # Add vertical line to separate clusters
            fig.add_shape(
                type="line",
                x0=current_pos
                - 0.5,  # Position the line at the boundary between clusters
                x1=current_pos - 0.5,  # Vertical line (same x position)
                y0=-0.5,  # Start of the line (bottom)
                y1=len(corr_matrix) - 0.5,  # End of the line (top)
                line=dict(color="white", width=2),  # White line with specified width
            )
            # Add horizontal line to separate clusters
            fig.add_shape(
                type="line",
                x0=-0.5,  # Start of the line (left)
                x1=len(corr_matrix) - 0.5,  # End of the line (right)
                y0=current_pos
                - 0.5,  # Position the line at the boundary between clusters
                y1=current_pos - 0.5,  # Horizontal line (same y position)
                line=dict(color="white", width=2),  # White line with specified width
            )
        current_pos += cluster_size  # Update position for the next cluster

    # Step 5: Update layout for the figure
    fig.update_layout(
        title="Stock Correlation Matrix (Clustered)",  # Title for the plot
        xaxis_title="Stocks",  # Label for the x-axis
        yaxis_title="Stocks",  # Label for the y-axis
        width=800,  # Width of the plot
        height=800,  # Height of the plot
        template="plotly_dark",  # Dark theme for the plot
    )

    # Step 6: Return the figure object (the correlation heatmap)
    return fig


def create_cluster_plot(returns_data, clustering_results):
    """
    Create a scatter plot of stocks based on their first two principal components.
    The plot displays how stocks are grouped into clusters based on their return patterns,
    with exemplars (cluster centers) highlighted.
    """

    # Step 1: Prepare the returns data for visualization
    # Standardize the returns data before applying PCA to ensure each feature (stock) contributes equally
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(returns_data)

    # Step 2: Apply PCA for dimensionality reduction to 2 components
    # PCA reduces the dimensionality of the data, allowing us to plot stocks in a 2D space
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(
        scaled_returns.T  # Transpose to ensure we transform stocks, not timestamps
    )

    # Step 3: Create a DataFrame for plotting
    # The DataFrame holds the principal components for each stock, as well as cluster and exemplar information
    plot_df = pd.DataFrame(
        pca_results, columns=["PC1", "PC2"], index=returns_data.columns
    )
    plot_df["Cluster"] = clustering_results["labels"]  # Add cluster labels
    plot_df["Exemplar"] = plot_df.index.isin(
        clustering_results["exemplars"]
    )  # Mark exemplars

    # Step 4: Create the scatter plot using Plotly Express
    # The scatter plot shows the first two principal components (PC1, PC2) on the axes, colored by cluster
    fig = px.scatter(
        plot_df,
        x="PC1",  # Principal component 1 on x-axis
        y="PC2",  # Principal component 2 on y-axis
        color="Cluster",  # Color points by their cluster
        text=plot_df.index,  # Add stock tickers as text labels
        title="Stock Clusters based on Return Patterns",  # Title for the plot
        template="plotly_dark",  # Use the dark theme for the plot
    )

    # Step 5: Highlight the exemplars (cluster centers)
    # Exemplars are the "center" stocks of each cluster, and we mark them with a star shape
    exemplar_df = plot_df[plot_df["Exemplar"]]  # Filter to get the exemplars
    fig.add_trace(
        go.Scatter(
            x=exemplar_df["PC1"],  # PC1 for exemplars on x-axis
            y=exemplar_df["PC2"],  # PC2 for exemplars on y-axis
            mode="markers",  # Display as markers
            marker=dict(
                symbol="star", size=15, line=dict(color="white", width=2)
            ),  # Star symbol with white border
            name="Cluster Centers",  # Name for the legend
            showlegend=True,  # Display this trace in the legend
        )
    )

    # Step 6: Update the trace settings for better display
    fig.update_traces(
        textposition="top center", marker=dict(size=10)
    )  # Position text labels and set marker size

    # Step 7: Update layout settings for the plot
    fig.update_layout(
        width=800, height=600, showlegend=True
    )  # Set plot size and show the legend

    # Step 8: Return the figure
    return fig
