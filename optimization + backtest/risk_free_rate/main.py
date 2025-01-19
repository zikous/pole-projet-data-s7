def get_risk_free_rate(start_date, end_date, default_rate=0.02):
    """
    Extrait le taux sans risque moyen à partir des rendements du Trésor américain (symbolisé par ^TNX sur Yahoo Finance).
    Si les données ne sont pas disponibles ou en cas d'erreur, retourne un taux par défaut.

    Args:
        start_date (str): Date de début au format 'AAAA-MM-JJ'.
        end_date (str): Date de fin au format 'AAAA-MM-JJ'.
        default_rate (float): Taux par défaut à retourner en cas d'échec (par défaut 0.02, soit 2%).

    Returns:
        float: Taux sans risque moyen sous forme décimale (ex: 0.05 pour 5%).
    """
    try:
        # Importation de yfinance pour récupérer les données
        import yfinance as yf

        # Téléchargement des données du Trésor américain (^TNX représente l'indice des taux d'intérêt)
        data = yf.download("^TNX", start=start_date, end=end_date)

        # Si les données sont vides, retourne le taux par défaut
        if data.empty:
            return default_rate

        # Calcul de la moyenne des taux de clôture et conversion en pourcentage
        # Note : Les taux sont donnés en pourcentage (ex: 5.0 pour 5%), donc on divise par 100
        return float(data["Close"].mean().iloc[0]) / 100

    except Exception as e:
        # En cas d'erreur (ex: problème de connexion, données indisponibles), retourne le taux par défaut
        print(f"Erreur lors de la récupération du taux sans risque : {e}")
        return default_rate


# Exemple d'utilisation
if __name__ == "__main__":
    # Période d'intérêt : de 2007 à 2009 (crise financière)
    start_date = "2007-01-01"
    end_date = "2009-01-01"

    # Appel de la fonction pour obtenir le taux sans risque
    risk_free_rate = get_risk_free_rate(start_date, end_date)

    # Affichage du résultat
    print(
        f"Le taux sans risque moyen entre {start_date} et {end_date} est de {risk_free_rate:.2%}"
    )
