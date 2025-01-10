import plotly.graph_objects as go
import json
import plotly


def create_price_chart(hist_data):
    """Create an interactive price chart using plotly"""
    # Create a candlestick chart with the historical data (Open, High, Low, Close)
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=hist_data.index,  # Date or time on the x-axis
                open=hist_data["Open"],  # Open price
                high=hist_data["High"],  # High price
                low=hist_data["Low"],  # Low price
                close=hist_data["Close"],  # Close price
            )
        ]
    )

    # Customize the layout of the chart
    fig.update_layout(
        title="Price History",  # Title of the chart
        yaxis_title="Price",  # Label for the y-axis
        xaxis_title="Date",  # Label for the x-axis
        template="plotly_dark",  # Dark theme for the chart
    )

    # Convert the figure to JSON format to send it via an API or embed it in a webpage
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def create_portfolio_chart(values, dates):
    """Create an interactive portfolio performance chart"""
    # Initialize a figure for the portfolio chart
    fig = go.Figure()

    # Loop through the strategies and their performance values
    for strategy, performance in values.items():
        # Add each strategy's performance as a line on the chart
        fig.add_trace(
            go.Scatter(
                x=dates,  # Dates for the x-axis
                y=performance,  # Performance values for the y-axis
                name=strategy.replace(
                    "_", " "
                ).title(),  # Strategy name (formatted nicely)
                mode="lines",  # Line chart mode
            )
        )

    # Customize the layout of the portfolio performance chart
    fig.update_layout(
        title="Portfolio Performance Comparison",  # Title of the chart
        yaxis_title="Portfolio Value ($)",  # Label for the y-axis
        xaxis_title="Date",  # Label for the x-axis
        template="plotly_dark",  # Dark theme for the chart
    )

    # Convert the portfolio performance chart to JSON format
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
