import plotly.graph_objects as go  # Importe Plotly pour créer des graphiques interactifs
import json  # Importe le module JSON pour sérialiser les graphiques
import plotly  # Importe Plotly pour l'encodage JSON des graphiques


def create_price_chart(hist_data):
    """
    Crée un graphique interactif de type chandelier (candlestick) à partir des données historiques.
    Args:
        hist_data (DataFrame): Données historiques contenant les colonnes Open, High, Low, Close.
    Returns:
        str: Graphique sérialisé en JSON pour une utilisation dans une API ou une page web.
    """
    # Crée un graphique en chandelier avec les données historiques
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=hist_data.index,  # Dates sur l'axe des x
                open=hist_data["Open"],  # Prix d'ouverture
                high=hist_data["High"],  # Prix le plus haut
                low=hist_data["Low"],  # Prix le plus bas
                close=hist_data["Close"],  # Prix de clôture
            )
        ]
    )

    # Personnalise la mise en page du graphique
    fig.update_layout(
        title="Historique des Prix",  # Titre du graphique
        yaxis_title="Prix",  # Titre de l'axe des y
        xaxis_title="Date",  # Titre de l'axe des x
        template="plotly_dark",  # Thème sombre pour le graphique
    )

    # Convertit le graphique en JSON pour l'envoyer via une API ou l'intégrer dans une page web
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def create_portfolio_chart(values, dates):
    """
    Crée un graphique interactif pour comparer les performances de plusieurs stratégies de portefeuille.
    Args:
        values (dict): Dictionnaire contenant les performances des stratégies (clé = nom de la stratégie, valeur = liste des valeurs).
        dates (list): Liste des dates correspondant aux performances.
    Returns:
        str: Graphique sérialisé en JSON pour une utilisation dans une API ou une page web.
    """
    # Initialise une figure pour le graphique de performance du portefeuille
    fig = go.Figure()

    # Parcourt chaque stratégie et ses performances
    for strategy, performance in values.items():
        # Ajoute une ligne pour chaque stratégie
        fig.add_trace(
            go.Scatter(
                x=dates,  # Dates sur l'axe des x
                y=performance,  # Performances sur l'axe des y
                name=strategy.replace(
                    "_", " "
                ).title(),  # Nom de la stratégie (formaté)
                mode="lines",  # Mode ligne pour le graphique
            )
        )

    # Personnalise la mise en page du graphique
    fig.update_layout(
        title="Comparaison des Performances du Portefeuille",  # Titre du graphique
        yaxis_title="Valeur du Portefeuille ($)",  # Titre de l'axe des y
        xaxis_title="Date",  # Titre de l'axe des x
        template="plotly_dark",  # Thème sombre pour le graphique
    )

    # Convertit le graphique en JSON pour l'envoyer via une API ou l'intégrer dans une page web
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
