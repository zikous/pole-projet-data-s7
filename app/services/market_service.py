import yfinance as yf  # Importe yfinance pour télécharger les données boursières
import pandas as pd  # Importe pandas pour manipuler les données
import plotly.graph_objects as go  # Importe Plotly pour créer des graphiques interactifs
import plotly.express as px  # Importe Plotly Express pour des graphiques simplifiés
import numpy as np  # Importe numpy pour les calculs numériques
from datetime import datetime  # Importe datetime pour gérer les dates
import logging  # Importe logging pour la journalisation des erreurs
from sklearn.cluster import (
    AffinityPropagation,
)  # Importe Affinity Propagation pour le clustering
from sklearn.preprocessing import (
    StandardScaler,
)  # Importe StandardScaler pour normaliser les données
from sklearn.decomposition import (
    PCA,
)  # Importe PCA pour la réduction de dimensionnalité

# Initialise le logger pour ce module
logger = logging.getLogger(__name__)  # Crée un logger avec le nom du module


def convert_to_serializable(obj):
    """
    Convertit les objets numpy/pandas en types sérialisables en JSON.
    Args:
        obj: L'objet à convertir (numpy, pandas, etc.).
    Returns:
        L'objet converti en un type sérialisable (int, float, dict, list, etc.).
    """
    # Si l'objet est un entier numpy, le convertit en entier Python
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)

    # Si l'objet est un float numpy, le convertit en float Python
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)

    # Si l'objet est un DataFrame pandas, convertit chaque colonne en dictionnaire
    elif isinstance(obj, pd.DataFrame):
        return {str(k): convert_to_serializable(v) for k, v in obj.to_dict().items()}

    # Si l'objet est un dictionnaire, convertit chaque clé-valeur
    elif isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}

    # Si l'objet est une liste, un tuple ou un ndarray numpy, convertit chaque élément
    elif isinstance(obj, (list, tuple, np.ndarray)):
        return [convert_to_serializable(x) for x in obj]

    # Si l'objet est un Timestamp pandas ou un datetime, le convertit en chaîne ISO
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()

    # Pour les autres types, retourne l'objet tel quel (déjà sérialisable)
    return obj


def get_correlation_and_clusters(tickers, start_date, end_date):
    """
    Récupère les données boursières, calcule la matrice de corrélation et effectue un clustering.
    Args:
        tickers (list): Liste des symboles boursiers à analyser.
        start_date (str): Date de début pour les données historiques.
        end_date (str): Date de fin pour les données historiques.
    Returns:
        dict: Résultats de l'analyse (matrice de corrélation, clusters, graphiques, etc.).
    """
    try:
        # Étape 1 : Nettoie et valide les tickers (supprime les espaces et met en majuscules)
        tickers = [ticker.strip().upper() for ticker in tickers]

        # Initialise les structures de données pour stocker les prix et les rendements
        data = pd.DataFrame()
        returns_data = pd.DataFrame()

        # Liste pour suivre les tickers qui ont échoué
        failed_tickers = []

        # Étape 2 : Télécharge les données historiques pour chaque ticker
        for ticker in tickers:
            try:
                # Récupère les données historiques avec yfinance
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date)

                # Si aucune donnée n'est disponible, ajoute le ticker à la liste des échecs
                if hist.empty:
                    failed_tickers.append(ticker)
                    continue

                # Stocke le prix de clôture et calcule les rendements quotidiens
                data[ticker] = hist["Close"]
                returns_data[ticker] = hist["Close"].pct_change().fillna(0)
            except Exception as e:
                # Log les erreurs pour les tickers qui ont échoué
                logger.error(
                    f"Erreur lors de la récupération des données pour {ticker}: {str(e)}"
                )
                failed_tickers.append(ticker)

        # Étape 3 : Vérifie si tous les tickers ont échoué
        if len(failed_tickers) == len(tickers):
            return {
                "success": False,
                "error": "Impossible de récupérer les données pour les tickers fournis",
            }

        # Étape 4 : Log un avertissement si certains tickers ont échoué
        if failed_tickers:
            logger.warning(
                f"Échec de récupération des données pour les tickers : {failed_tickers}"
            )

        # Étape 5 : Supprime les tickers échoués de la liste pour le traitement ultérieur
        tickers = [t for t in tickers if t not in failed_tickers]

        # Étape 6 : Calcule la matrice de corrélation pour les prix de clôture
        corr_matrix = data.corr()

        # Étape 7 : Effectue le clustering sur les rendements
        clustering_results = perform_clustering(returns_data)

        # Étape 8 : Crée les graphiques pour la matrice de corrélation et les clusters
        corr_plot = create_correlation_plot(corr_matrix, clustering_results)
        cluster_plot = create_cluster_plot(returns_data, clustering_results)

        # Étape 9 : Convertit les résultats en format JSON pour la réponse de l'API
        result = {
            "success": True,
            "correlation_matrix": convert_to_serializable(
                corr_matrix
            ),  # Matrice de corrélation
            "correlation_plot": corr_plot.to_json(),  # Graphique de corrélation
            "cluster_plot": cluster_plot.to_json(),  # Graphique des clusters
            "clusters": convert_to_serializable(
                clustering_results
            ),  # Résultats du clustering
            "failed_tickers": failed_tickers,  # Tickers ayant échoué
        }

        return result

    except Exception as e:
        # Log les erreurs inattendues et retourne une réponse d'erreur
        logger.error(f"Erreur lors de l'analyse : {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}


def perform_clustering(returns_data):
    """
    Effectue un clustering par Affinity Propagation sur les rendements boursiers.
    Args:
        returns_data (DataFrame): Données des rendements quotidiens.
    Returns:
        dict: Résultats du clustering (labels, nombre de clusters, exemplaires, etc.).
    """
    # Étape 1 : Normalise les données de rendement pour le clustering
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(returns_data)

    # Étape 2 : Applique l'algorithme Affinity Propagation
    af = AffinityPropagation(random_state=42, damping=0.9)  # damping pour la stabilité
    cluster_labels = af.fit_predict(scaled_returns.T)  # Clustering par actions

    # Étape 3 : Prépare les résultats du clustering
    clustering_results = {
        "labels": [int(x) for x in cluster_labels],  # Labels des clusters
        "n_clusters": int(len(set(cluster_labels))),  # Nombre de clusters
        "clusters": {},  # Dictionnaire pour stocker les actions par cluster
        "exemplars": [],  # Liste des exemplaires (centres des clusters)
    }

    # Étape 4 : Groupe les actions par cluster
    for i, ticker in enumerate(returns_data.columns):
        cluster_num = int(cluster_labels[i])
        if cluster_num not in clustering_results["clusters"]:
            clustering_results["clusters"][cluster_num] = []
        clustering_results["clusters"][cluster_num].append(
            str(ticker)
        )  # Ajoute l'action au cluster

        # Étape 5 : Identifie les exemplaires (centres des clusters)
        if i in af.cluster_centers_indices_:
            clustering_results["exemplars"].append(str(ticker))  # Ajoute l'exemplaire

    # Étape 6 : Retourne les résultats du clustering
    return clustering_results


def create_correlation_plot(corr_matrix, clustering_results):
    """
    Crée une heatmap de corrélation avec les clusters.
    Args:
        corr_matrix (DataFrame): Matrice de corrélation.
        clustering_results (dict): Résultats du clustering.
    Returns:
        go.Figure: Heatmap de corrélation avec séparateurs de clusters.
    """
    # Étape 1 : Trie les tickers par cluster
    sorted_tickers = []
    for cluster in sorted(clustering_results["clusters"].keys()):
        sorted_tickers.extend(clustering_results["clusters"][cluster])

    # Étape 2 : Réorganise la matrice de corrélation selon l'ordre des clusters
    corr_matrix = corr_matrix.loc[sorted_tickers, sorted_tickers]

    # Étape 3 : Crée la heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,  # Valeurs de corrélation
            x=corr_matrix.index,  # Tickers sur l'axe x
            y=corr_matrix.columns,  # Tickers sur l'axe y
            hoverongaps=False,  # Désactive le survol des cellules vides
            colorscale="RdBu",  # Échelle de couleurs (rouge/bleu)
            zmin=-1,  # Valeur minimale de corrélation
            zmax=1,  # Valeur maximale de corrélation
            text=np.round(
                corr_matrix.values, decimals=2
            ),  # Affiche les valeurs de corrélation
            texttemplate="%{text}",  # Format du texte
            textfont={"size": 10},  # Taille de la police
            showscale=True,  # Affiche l'échelle de couleurs
        )
    )

    # Étape 4 : Ajoute des séparateurs de clusters
    current_pos = 0
    for cluster in sorted(clustering_results["clusters"].keys()):
        cluster_size = len(clustering_results["clusters"][cluster])
        if current_pos > 0:
            # Ajoute une ligne verticale
            fig.add_shape(
                type="line",
                x0=current_pos - 0.5,
                x1=current_pos - 0.5,
                y0=-0.5,
                y1=len(corr_matrix) - 0.5,
                line=dict(color="white", width=2),
            )
            # Ajoute une ligne horizontale
            fig.add_shape(
                type="line",
                x0=-0.5,
                x1=len(corr_matrix) - 0.5,
                y0=current_pos - 0.5,
                y1=current_pos - 0.5,
                line=dict(color="white", width=2),
            )
        current_pos += cluster_size  # Met à jour la position pour le prochain cluster

    # Étape 5 : Met à jour la mise en page de la figure
    fig.update_layout(
        title="Matrice de Corrélation des Actions (Clusters)",  # Titre
        xaxis_title="Actions",  # Titre de l'axe x
        yaxis_title="Actions",  # Titre de l'axe y
        width=800,  # Largeur
        height=800,  # Hauteur
        template="plotly_dark",  # Thème sombre
    )

    # Étape 6 : Retourne la figure
    return fig


def create_cluster_plot(returns_data, clustering_results):
    """
    Crée un graphique des clusters basé sur les deux premières composantes principales.
    Args:
        returns_data (DataFrame): Données des rendements.
        clustering_results (dict): Résultats du clustering.
    Returns:
        go.Figure: Graphique des clusters avec les exemplaires mis en évidence.
    """
    # Étape 1 : Normalise les données de rendement
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(returns_data)

    # Étape 2 : Applique PCA pour réduire la dimensionnalité à 2 composantes
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(scaled_returns.T)  # Clustering par actions

    # Étape 3 : Crée un DataFrame pour le graphique
    plot_df = pd.DataFrame(
        pca_results, columns=["PC1", "PC2"], index=returns_data.columns
    )
    plot_df["Cluster"] = clustering_results["labels"]  # Ajoute les labels des clusters
    plot_df["Exemplar"] = plot_df.index.isin(
        clustering_results["exemplars"]
    )  # Marque les exemplaires

    # Étape 4 : Crée le graphique des clusters avec Plotly Express
    fig = px.scatter(
        plot_df,
        x="PC1",  # Composante principale 1
        y="PC2",  # Composante principale 2
        color="Cluster",  # Couleur par cluster
        text=plot_df.index,  # Tickers comme étiquettes
        title="Clusters d'Actions basés sur les Rendements",  # Titre
        template="plotly_dark",  # Thème sombre
    )

    # Étape 5 : Met en évidence les exemplaires (centres des clusters)
    exemplar_df = plot_df[plot_df["Exemplar"]]
    fig.add_trace(
        go.Scatter(
            x=exemplar_df["PC1"],
            y=exemplar_df["PC2"],
            mode="markers",
            marker=dict(symbol="star", size=15, line=dict(color="white", width=2)),
            name="Centres des Clusters",
            showlegend=True,
        )
    )

    # Étape 6 : Met à jour les paramètres des traces
    fig.update_traces(textposition="top center", marker=dict(size=10))

    # Étape 7 : Met à jour la mise en page
    fig.update_layout(width=800, height=600, showlegend=True)

    # Étape 8 : Retourne la figure
    return fig
