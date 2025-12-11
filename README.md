# Projet de Régression Linéaire sur les Prix de Véhicules d'Occasion

## Introduction

Ce projet analyse un ensemble de données sur les véhicules d'occasion afin de construire un modèle de régression linéaire capable de prédire leur prix. Le notebook Jupyter `Regression_linéaire.ipynb` contient l'ensemble du processus, du nettoyage des données à l'évaluation du modèle. Une application Streamlit (`app.py`) a également été développée pour interagir avec le modèle final.

## Structure du Projet

- **`Regression_linéaire.ipynb`**: Le notebook principal qui détaille toutes les étapes de l'analyse, y compris le nettoyage des données, la visualisation et la modélisation.
- **`app.py`**: Une application web simple construite avec Streamlit qui charge le modèle de régression entraîné et permet de faire des prédictions de prix en fonction des caractéristiques du véhicule.
- **`requirements.txt`**: La liste des dépendances Python nécessaires pour exécuter le projet.
- **`.png` files**: Divers graphiques générés durant l'analyse exploratoire des données, montrant les relations entre différentes variables et le prix.
- **`.pkl` files**:
  - `linear_regression_model.pkl`: Le modèle de régression linéaire sauvegardé après entraînement.
  - `model_columns.pkl`: La liste des colonnes du modèle, nécessaire pour s'assurer que les données d'entrée de l'application correspondent à celles utilisées pour l'entraînement.

## Méthodologie

Le projet suit les étapes suivantes :

1.  **Nettoyage des données**: Chargement du jeu de données, suppression des colonnes inutiles, gestion des valeurs manquantes et des doublons.
2.  **Analyse Exploratoire des Données (EDA)**: Création de visualisations (histogrammes, nuages de points, etc.) pour comprendre les distributions et les relations entre les variables.
3.  **Feature Engineering**: Création de nouvelles variables pertinentes, comme l'âge du véhicule, pour améliorer la performance du modèle.
4.  **Modélisation**: Entraînement d'un modèle de régression linéaire pour prédire le prix des véhicules.
5.  **Évaluation du modèle**: Analyse des performances du modèle à l'aide de métriques comme le R² et le RMSE, ainsi qu'une analyse des résidus.

## Comment Exécuter le Projet

Pour exécuter l'application Streamlit, suivez ces étapes :

1.  **Installez les dépendances**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Exécutez l'application**:
    ```bash
    streamlit run app.py
    ```

## Génération des Fichiers `.pkl`

Les fichiers `linear_regression_model.pkl` et `model_columns.pkl` sont générés à la fin du notebook `Regression_linéaire.ipynb`. Si ces fichiers sont manquants, vous devez exécuter l'intégralité du notebook pour les créer. Ces fichiers sont essentiels au fonctionnement de l'application `app.py`.

## Résumé des Résultats

Le modèle de régression linéaire atteint un **R² de 0.62**, ce qui signifie qu'il explique environ 62% de la variance des prix des véhicules. L'erreur moyenne (RMSE) est d'environ **7932 $**.

Les facteurs les plus influents sur le prix sont :
- **L'âge du véhicule** (`car_age`) et le **kilométrage** (`odometer`), qui ont un impact négatif significatif sur le prix.
- Le **type de véhicule** (par exemple, les "pickup" et les "trucks" ont tendance à être plus chers).
- Le **carburant** a également un impact notable, les véhicules à essence, hybrides ou électriques ayant des prix de base différents.
