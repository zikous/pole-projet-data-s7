# Projet Data Science Finance : Modèles Graphiques Parcimonieux et Optimisation de Portefeuille

Ce projet combine des techniques avancées de finance, d'analyse de données et d'intelligence artificielle pour l'optimisation de portefeuilles financiers et l'analyse des sentiments à l'aide de FinBERT.

## Fonctionnalités Principales

- **Optimisation de Portefeuille** : Maximisation du ratio de Sharpe, minimisation des risques et simulations Monte Carlo.
- **Analyse des Sentiments** : Utilisation de FinBERT pour classifier les sentiments des nouvelles financières en positif, neutre et négatif.
- **Backtesting** : Validation des stratégies d'investissement sur des données historiques.
- **Application Web** : Interface interactive pour visualiser les analyses et optimiser les portefeuilles.

## Technologie Utilisée

- **Backend** : Flask (Python)
- **Frontend** : HTML5, CSS (Tailwind CSS), JavaScript
- **Bibliothèques** : yfinance, pandas, numpy, FinBERT, matplotlib, seaborn

## Structure du Répertoire

Le projet est organisé en plusieurs dossiers pour une meilleure gestion des différentes parties du projet :

- **app** : Contient l'application web Flask, y compris les templates HTML, les fichiers CSS/JS, et les routes pour l'interface utilisateur.
- **distribution + stationarity** : Comprend des analyses statistiques sur la distribution des données et la stationnarité des séries temporelles.
- **optimization + backtest** : Contient les scripts pour l'optimisation de portefeuille et les expérimentations de backtesting.
- **random forest approach** : Inclut les notebooks et scripts pour l'approche Random Forest dans l'analyse des données.
- **sentiment** : Dédié à l'analyse des sentiments avec FinBERT, y compris le prétraitement des données et la classification.

## Guide d'Utilisation

1. Clonez le dépôt :
   ```bash
   git clone https://github.com/zikous/pole-projet-data-s7
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
