import os  # Importe le module os pour interagir avec le système de fichiers
from functools import (
    lru_cache,
)  # Importe lru_cache pour la mise en cache des résultats de fonction
from transformers import (
    pipeline,
)  # Importe pipeline pour utiliser des modèles de NLP pré-entraînés
import yfinance as yf  # Importe yfinance pour récupérer les données boursières
from services.chart_service import (
    create_price_chart,
)  # Importe la fonction pour créer des graphiques de prix
import re  # Importe re pour les expressions régulières
from bs4 import (
    BeautifulSoup,
    SoupStrainer,
)  # Importe BeautifulSoup pour le parsing HTML
import requests  # Importe requests pour faire des requêtes HTTP
import nltk  # Importe nltk pour le traitement du langage naturel
from nltk.corpus import stopwords  # Importe stopwords pour filtrer les mots vides
from concurrent.futures import (
    ThreadPoolExecutor,
)  # Importe ThreadPoolExecutor pour le multithreading
from typing import Dict, List, Tuple, Optional  # Importe les types pour les annotations

# Constantes
USE_GPU = True  # Définit si le GPU doit être utilisé (False pour CPU)
MODEL_NAME = "yiyanghkust/finbert-tone"  # Nom du modèle de sentiment FinBERT
MODEL_DIR = "./models/finbert-tone"  # Répertoire pour sauvegarder le modèle localement
MAX_TEXT_LENGTH = 500  # Longueur maximale du texte pour l'analyse de sentiment
REQUEST_TIMEOUT = 10  # Temps d'attente pour les requêtes HTTP
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; Python-requests/2.28.1)"
}  # En-têtes pour les requêtes HTTP

# Initialisation de NLTK
try:
    nltk.download("stopwords", quiet=True)  # Télécharge les mots vides en silence
    stop_words = set(stopwords.words("english"))  # Charge les mots vides en anglais
except Exception as e:
    print(f"Erreur lors du téléchargement des mots vides : {str(e)}")
    stop_words = set()  # Utilise un ensemble vide en cas d'erreur

# Compilation des motifs regex
SPECIAL_CHARS_PATTERN = re.compile(r"[^\w\s]")  # Pour supprimer les caractères spéciaux
NUMBERS_PATTERN = re.compile(r"\d+")  # Pour supprimer les chiffres


def clean_text(text: str) -> str:
    """
    Nettoie le texte en supprimant les caractères spéciaux et les mots vides.
    Args:
        text (str): Texte à nettoyer.
    Returns:
        str: Texte nettoyé.
    """
    if not text:
        return ""

    try:
        text = SPECIAL_CHARS_PATTERN.sub(
            " ", text.lower()
        )  # Supprime les caractères spéciaux et met en minuscules
        text = NUMBERS_PATTERN.sub(" ", text)  # Supprime les chiffres
        words = [
            word
            for word in text.split()
            if word
            and word not in stop_words
            and len(word) > 2  # Filtre les mots vides et courts
        ]
        return " ".join(words)  # Rejoint les mots en une seule chaîne
    except Exception as e:
        print(f"Erreur lors du nettoyage du texte : {str(e)}")
        return ""


def truncate_text(text: str, max_length: int = MAX_TEXT_LENGTH) -> str:
    """
    Tronque le texte à une longueur maximale en respectant les limites des mots.
    Args:
        text (str): Texte à tronquer.
        max_length (int): Longueur maximale du texte.
    Returns:
        str: Texte tronqué.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length].rsplit(" ", 1)[
        0
    ]  # Tronque au dernier espace avant la limite


def load_or_download_model():
    """
    Charge le modèle localement s'il existe, sinon le télécharge.
    Returns:
        pipeline: Modèle de sentiment FinBERT.
    """
    try:
        os.makedirs(
            MODEL_DIR, exist_ok=True
        )  # Crée le répertoire du modèle s'il n'existe pas

        # Définit l'appareil (GPU ou CPU)
        device = 0 if USE_GPU else -1  # 0 pour GPU, -1 pour CPU

        # Si le modèle existe localement, essaie de le charger
        if os.path.exists(MODEL_DIR) and os.path.isfile(
            os.path.join(MODEL_DIR, "config.json")
        ):
            try:
                return pipeline("sentiment-analysis", model=MODEL_DIR, device=device)
            except Exception as e:
                print(f"Erreur lors du chargement du modèle local : {str(e)}")
                print("Tentative de retéléchargement du modèle...")

        # Télécharge et sauvegarde le modèle si le chargement local échoue
        model = pipeline("sentiment-analysis", model=MODEL_NAME, device=device)
        model.save_pretrained(MODEL_DIR)
        return model

    except Exception as e:
        print(f"Erreur dans load_or_download_model : {str(e)}")
        raise


# Instance globale du modèle
finbert_sentiment = None
try:
    finbert_sentiment = load_or_download_model()  # Charge ou télécharge le modèle
except Exception as e:
    print(f"Échec du chargement du modèle de sentiment : {str(e)}")


def fetch_article_content(url: str) -> Optional[str]:
    """
    Récupère le contenu d'un article avec gestion des erreurs et timeout.
    Args:
        url (str): URL de l'article.
    Returns:
        Optional[str]: Contenu de l'article ou None en cas d'erreur.
    """
    try:
        with requests.get(url, timeout=REQUEST_TIMEOUT, headers=HEADERS) as response:
            if response.status_code != 200:  # Vérifie si la requête a réussi
                return None

            # Parse le contenu de l'article à partir des balises <p>
            soup = BeautifulSoup(
                response.content, "html.parser", parse_only=SoupStrainer("p")
            )
            paragraphs = [
                p.get_text() for p in soup.find_all("p") if p.get_text().strip()
            ]

            if paragraphs:
                return " ".join(
                    paragraphs
                )  # Rejoint les paragraphes en une seule chaîne
            else:
                return None

    except Exception as e:
        print(f"Erreur lors de la récupération du contenu de l'article : {str(e)}")
        return None


def process_news_item(item: Dict) -> Optional[Dict]:
    """
    Traite un élément de news avec analyse de sentiment.
    Args:
        item (Dict): Élément de news contenant un titre et une URL.
    Returns:
        Optional[Dict]: Résultat de l'analyse de sentiment ou None en cas d'erreur.
    """
    global finbert_sentiment

    try:
        # Vérifie que l'élément de news est valide
        if (
            not isinstance(item, dict)
            or "content" not in item
            or "title" not in item["content"]
            or "canonicalUrl" not in item["content"]
        ):
            print(f"Format d'élément de news invalide : {item}")
            return None

        title = item["content"]["title"]  # Titre de l'article
        link = item["content"]["canonicalUrl"]["url"]  # URL de l'article

        # Récupère le contenu de l'article
        content = fetch_article_content(link)
        print(f"Contenu récupéré : {content}")

        # Combine le titre et le contenu (si disponible) pour l'analyse de sentiment
        sentiment_text = f"{title} {content}" if content else title
        cleaned_text = clean_text(sentiment_text)  # Nettoie le texte
        print(f"Texte nettoyé : {cleaned_text}")

        truncated_text = truncate_text(cleaned_text)  # Tronque le texte
        print(f"Texte tronqué : {truncated_text}")

        if not truncated_text:
            print("Aucun texte après troncature, analyse de sentiment ignorée.")
            return None

        # Recharge le modèle s'il n'est pas chargé
        if finbert_sentiment is None:
            print("Le modèle n'est pas chargé !")
            finbert_sentiment = load_or_download_model()

        # Effectue l'analyse de sentiment
        sentiment = finbert_sentiment(truncated_text)
        print(f"Résultat de l'analyse de sentiment : {sentiment}")

        return {
            "title": title,
            "link": link,
            "sentiment": sentiment[0][
                "label"
            ],  # Étiquette de sentiment (positif/négatif/neutre)
            "score": float(sentiment[0]["score"]),  # Score de confiance
        }

    except Exception as e:
        print(f"Erreur lors du traitement de l'élément de news : {str(e)}")
        return None


def get_stock_data(ticker: str) -> Tuple[Dict, object, object, List[Dict]]:
    """
    Récupère les données boursières avec gestion améliorée des erreurs.
    Args:
        ticker (str): Symbole boursier (ex: "AAPL").
    Returns:
        Tuple[Dict, object, object, List[Dict]]: Informations de base, historique, graphique et news.
    """
    try:
        stock = yf.Ticker(ticker)  # Récupère les données du ticker

        # Récupère les informations et l'historique de manière concurrente
        with ThreadPoolExecutor() as executor:
            info_future = executor.submit(lambda: stock.info)
            hist_future = executor.submit(lambda: stock.history(period="1y"))

            info = info_future.result()  # Informations de base
            hist = hist_future.result()  # Historique des prix

        # Traite les informations de base
        basic_info = {
            "name": info.get("longName", "N/A"),  # Nom de l'entreprise
            "sector": info.get("sector", "N/A"),  # Secteur d'activité
            "market_cap": (
                float(info.get("marketCap", 0)) if info.get("marketCap") else "N/A"
            ),  # Capitalisation boursière
            "pe_ratio": (
                float(info.get("trailingPE", 0)) if info.get("trailingPE") else "N/A"
            ),  # Ratio P/E
            "dividend_yield": (
                float(info.get("dividendYield", 0))
                if info.get("dividendYield")
                else "N/A"
            ),  # Rendement du dividende
            "beta": float(info.get("beta", 0)) if info.get("beta") else "N/A",  # Bêta
            "profit_margin": (
                float(info.get("profitMargins", 0))
                if info.get("profitMargins")
                else "N/A"
            ),  # Marge bénéficiaire
            "debt_to_equity": (
                float(info.get("debtToEquity", 0))
                if info.get("debtToEquity")
                else "N/A"
            ),  # Ratio dette/capitaux propres
            "roe": (
                float(info.get("returnOnEquity", 0))
                if info.get("returnOnEquity")
                else "N/A"
            ),  # Retour sur capitaux propres
            "roa": (
                float(info.get("returnOnAssets", 0))
                if info.get("returnOnAssets")
                else "N/A"
            ),  # Retour sur actifs
        }

        # Calcule les moyennes mobiles si l'historique est disponible
        if not hist.empty:
            hist["SMA_50"] = (
                hist["Close"].rolling(window=50).mean()
            )  # Moyenne mobile sur 50 jours
            hist["SMA_200"] = (
                hist["Close"].rolling(window=200).mean()
            )  # Moyenne mobile sur 200 jours

        # Crée un graphique des prix
        price_chart = create_price_chart(hist)

        # Traite les news associées au ticker
        stock_news = getattr(stock, "news", []) or []  # Récupère les news
        news = [
            process_news_item(item) for item in stock_news if process_news_item(item)
        ]  # Analyse les news

        print(f"Nombre d'éléments de news traités avec succès : {len(news)}")
        return basic_info, hist, price_chart, news

    except Exception as e:
        print(f"Erreur dans get_stock_data : {str(e)}")
        raise
