# Projet Data Science Finance : Modèles Graphiques Parcimonieux et Optimisation de Portefeuille

Ce projet combine des techniques avancées de finance, d'analyse de données et d'intelligence artificielle pour l'optimisation de portefeuilles financiers et l'analyse des sentiments à l'aide de FinBERT.

## Contenu du Dépôt

```
├── app
│   ├── models
│   │   └── finbert-tone
│   ├── routes
│   │   └── __pycache__
│   ├── services
│   │   └── __pycache__
│   ├── templates
│   └── __pycache__
├── models
│   └── finbert-tone
├── optimization + backtest
│   ├── examples
│   ├── notebooks
│   ├── risk_free_rate
│   └── src
│       ├── config
│       │   └── __pycache__
│       ├── data
│       │   └── __pycache__
│       ├── optimization
│       │   └── __pycache__
│       └── simulation
│           └── __pycache__
└── sentiment
```

## Fonctionnalités Principales

- **Optimisation de Portefeuille** : Maximisation du ratio de Sharpe, minimisation des risques et simulations Monte Carlo.
- **Analyse des Sentiments** : Utilisation de FinBERT pour classifier les sentiments des nouvelles financières en positif, neutre et négatif.
- **Backtesting** : Validation des stratégies d'investissement sur des données historiques.
- **Application Web** : Interface interactive pour visualiser les analyses et optimiser les portefeuilles.

## Technologie Utilisée

- **Backend** : Flask (Python)
- **Frontend** : HTML5, CSS (Tailwind CSS), JavaScript
- **Bibliothèques** : yfinance, pandas, numpy, FinBERT, matplotlib, seaborn

## Guide d'Utilisation

1. Clonez le dépôt :
   ```bash
   git clone <url_du_dépôt>
   ```
2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
3. Lancez l'application :
   ```bash
   cd app
   python app.py
   ```

## Auteurs

- Bheddar Zakaria
- Adib Aymane Chaoui
- Boumoussou Younes
- EL Achkar Salma
- Pilorget Maxime
- ELyazidi Nabil

Encadrant : Maxime Ferreira Da Costa
