# services/chart_service.py
import plotly.graph_objects as go
import json
import plotly


def create_price_chart(hist_data):
    """Create an interactive price chart using plotly"""
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=hist_data.index,
                open=hist_data["Open"],
                high=hist_data["High"],
                low=hist_data["Low"],
                close=hist_data["Close"],
            )
        ]
    )

    fig.update_layout(
        title="Price History",
        yaxis_title="Price",
        xaxis_title="Date",
        template="plotly_dark",
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def create_portfolio_chart(values, dates):
    """Create an interactive portfolio performance chart"""
    fig = go.Figure()

    for strategy, performance in values.items():
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=performance,
                name=strategy.replace("_", " ").title(),
                mode="lines",
            )
        )

    fig.update_layout(
        title="Portfolio Performance Comparison",
        yaxis_title="Portfolio Value ($)",
        xaxis_title="Date",
        template="plotly_dark",
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
