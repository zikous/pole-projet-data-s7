def get_risk_free_rate(start_date, end_date, default_rate=0.02):
    """
    Extrait le taux sans risque moyen à partir des rendements du Trésor
    Retourne un taux par défaut si les données sont indisponibles
    """
    try:
        import yfinance as yf

        data = yf.download("^TNX", start=start_date, end=end_date)
        if data.empty:
            return default_rate

        return data["Close"].mean() / 100

    except:
        return default_rate


if __name__ == "__main__":
    start_date = "2007-01-01"
    end_date = "2009-01-01"
    risk_free_rate = get_risk_free_rate(start_date, end_date)
    print(f"Le taux sans risque moyen est de {risk_free_rate:.2%}")
