# 📊 Dashboard Analytics des Ventes - Rossmann Stores

Bienvenue dans le **Dashboard Analytics des Ventes** pour les magasins Rossmann. Ce projet vise à fournir une application interactive pour la gestion des stocks et la prévision des ventes, répondant aux besoins des acteurs du secteur Sage X3. L'application combine un tableau de bord interactif avec un modèle prédictif robuste pour anticiper les ventes futures.

## 🔍 Table des Matières

1. [À propos](#à-propos)
2. [Fonctionnalités](#fonctionnalités)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Entraînement des Modèles](#entraînement-des-modèles)
6. [Structure du Projet](#structure-du-projet)
7. [Résultats et Benchmarking](#résultats-et-benchmarking)
8. [Contribution](#contribution)
9. [Licence](#licence)
10. [Contact](#contact)

## À propos

Le **Dashboard Analytics des Ventes** est une application Streamlit conçue pour permettre une visualisation interactive des données de ventes et de stocks des magasins Rossmann. En outre, l'application intègre plusieurs modèles de prévision des ventes pour anticiper les performances futures, facilitant ainsi la prise de décision stratégique.

## Fonctionnalités

### 🎨 Tableau de Bord Interactif
- **Vue d'ensemble** : Statistiques clés telles que les ventes totales, les ventes moyennes, le nombre total de clients, etc.
- **Ventes au Fil du Temps** : Visualisation des ventes quotidiennes et mensuelles.
- **Analyse par Type et Assortiment** : Ventes par type de magasin et par niveau d'assortiment.
- **Impact des Promotions et Jours Fériés** : Analyse des ventes avec et sans promotions, ainsi que pendant les jours fériés.
- **Distance à la Compétition** : Distribution de la distance entre les magasins Rossmann et leurs concurrents.
- **Analyse des Clients** : Nombre de clients quotidiens et par magasin.
- **Corrélations** : Matrice de corrélation entre les différentes variables.
- **Aperçu des Données** : Visualisation des premières lignes des données filtrées.

### 🔮 Prévisions des Ventes
- **Prévisions par Magasin** : Génération de prévisions de ventes pour un magasin sélectionné sur une période donnée.
- **Prévisions Globales** : Génération de prévisions de ventes agrégées pour l'ensemble des magasins sur une période donnée.
- **Indicateurs Clés** : Affichage des ventes totales prévisionnelles, des ventes moyennes quotidiennes et des ventes moyennes par magasin.
- **Visualisation des Prévisions** : Graphiques interactifs (courbes ou histogrammes) des prévisions de ventes.
- **Téléchargement des Prévisions** : Option pour télécharger les prévisions au format CSV.

### 📈 Modèles de Prévision
- **Régression Linéaire**
- **Forêt Aléatoire (Random Forest)**
- **XGBoost**
- **ARIMA/SARIMA**
- **Lissage Exponentiel (Holt-Winters)**
- **Prophet**

### 📊 Benchmarking des Modèles
- Tableau comparatif des performances des modèles avec des métriques telles que MAE, RMSE et MAPE.
- Sélection du meilleur modèle basé sur les performances évaluées.

## Installation

### Prérequis

- **Python 3.8+**
- **Git**

### Étapes d'Installation

1. **Cloner le Répertoire**

    ```bash
    git clone https://github.com/AbrahamKOLOBOE/Rossmann-Sales-Dashboard.git
    cd Rossmann-Sales-Dashboard
    ```

2. **Créer un Environnement Virtuel**

    ```bash
    python -m venv venv
    source venv/bin/activate  # Sur Windows : venv\Scripts\activate
    ```

3. **Installer les Dépendances**

    ```bash
    pip install -r requirements.txt
    ```

4. **Télécharger les Données**

    Assurez-vous d'avoir les fichiers suivants dans le répertoire `data/` :
    - `train.csv` : Données historiques de ventes.
    - `store.csv` : Informations complémentaires sur les magasins.

## Usage

### 1. Entraînement des Modèles

Avant de lancer le dashboard, vous devez entraîner les modèles de prévision. Vous pouvez le faire via le notebook Jupyter :

```bash
cd notebooks
jupyter notebook Train-models.ipynb
```

Ce notebook :
- Prétraite les données.
- Entraîne plusieurs modèles de prévision.
- Enregistre les modèles entraînés et les artefacts (graphiques, métriques).

### 2. Lancer le Dashboard

Une fois les modèles entraînés, lancez l'application Streamlit :

```bash
streamlit run Dashboard.py
```

### 3. Prévisions des Ventes

Vous pouvez accéder aux prévisions de ventes via la page séparée de prédiction :

```bash
streamlit run pages/01_🔮_Prévision_des_ventes.py
```

### 4. Naviguer dans le Dashboard

L'application est divisée en plusieurs onglets pour une meilleure organisation :

- **📋 Vue d'ensemble** : Statistiques clés.
- **📅 Ventes au Fil du Temps** : Visualisation des ventes quotidiennes et mensuelles.
- **🏪 Analyse par Type & Assortiment** : Analyse des ventes par type de magasin et assortiment.
- **🎁 Promotions & Jours Fériés** : Impact des promotions et jours fériés sur les ventes.
- **📍 Distance à la Compétition** : Analyse de la distance des magasins à leurs concurrents.
- **👥 Analyse des Clients** : Analyse du nombre de clients.
- **📊 Corrélations** : Matrice de corrélation des variables.
- **📄 Aperçu des Données** : Visualisation des données filtrées.
- **🔮 Prévisions des Ventes** : Prévisions pour un magasin spécifique.
- **🌐 Prévisions Globales** : Prévisions agrégées pour tous les magasins.

## Entraînement des Modèles

### 1. Entraînement avec Jupyter Notebook

Le notebook `notebooks/Train-models.ipynb` effectue les tâches suivantes :
- Chargement et prétraitement des données.
- Entraînement de plusieurs modèles de prévision.
- Évaluation des modèles et enregistrement des artefacts.
- Sélection du meilleur modèle basé sur les métriques d'évaluation.

### 2. Artefacts Enregistrés

Les artefacts générés comprennent :
- **Modèles Enregistrés** : Stockés dans le dossier `models/`.
- **Graphes et Logs** : Stockés dans le dossier `artefacts/`.
- **Tableau Comparatif des Modèles** : `artefacts/model_comparison.csv`.

## Structure du Projet

```plaintext
Rossmann-Sales-Dashboard/
│
├── artefacts/
│   ├── Linear_Regression_predictions.png
│   ├── model_comparison.csv
│   ├── model_performance.txt
│   ├── Random_Forest_predictions.png
│   └── XGBoost_predictions.png
│
├── data/
│   ├── store.csv
│   └── train.csv
│
├── models/
│   ├── holt_winters_model_store_1.joblib
│   ├── linear_regression_pipeline.joblib
│   ├── prophet_model_store_1.joblib
│   ├── random_forest_pipeline.joblib
│   ├── sarima_model_store_1.joblib
│   └── xgboost_pipeline.joblib
│
├── notebooks/
│   └── Train-models.ipynb
│
├── pages/
│   └── 01_🔮_Prévision_des_ventes.py
│
├── .gitignore
├── Dashboard.py
├── Readme.md
└── requirements.txt
```

- **artefacts/** : Contient les graphiques de performances des modèles, les logs, et le tableau comparatif.
- **data/** : Contient les fichiers de données nécessaires.
- **models/** : Stocke les modèles entraînés et les pipelines de prétraitement.
- **notebooks/** : Contient les notebooks Jupyter pour l'entraînement des modèles.
- **pages/** : Contient les pages supplémentaires pour Streamlit (prédictions).
- **Dashboard.py** : Script principal pour le tableau de bord Streamlit.
- **requirements.txt** : Liste des dépendances Python.
- **Readme.md** : Documentation du projet.

## Résultats et Benchmarking

### Tableau Comparatif des Modèles

Après l'entraînement, un tableau comparatif des performances des modèles est généré dans `artefacts/model_comparison.csv`. Ce tableau inclut les métriques suivantes :

| Modèle                  | MAE    | RMSE   | MAPE   |
|-|--|--|--|
| Régression Linéaire     | 1234.56| 2345.67| 12.34% |
| Forêt Aléatoire         | 987.65 | 1987.65| 9.87%  |
| XGBoost                 | 876.54 | 1876.54| 

8.76%  |
| SARIMA (Magasin 1)      | 1500.00| 2500.00| 15.00% |
| Holt-Winters (Magasin 1)| 1400.00| 2400.00| 14.00% |
| Prophet (Magasin 1)     | 1300.00| 2300.00| 13.00% |

*Les valeurs sont indicatives et dépendent des données d'entraînement réelles.*

### Benchmarking

Le benchmarking a été réalisé en comparant les métriques MAE, RMSE et MAPE pour chaque modèle. Le modèle avec les valeurs les plus basses pour ces métriques est considéré comme le meilleur modèle. Dans cet exemple, **XGBoost** présente les meilleures performances et est donc sélectionné pour les prévisions intégrées dans le dashboard.

## Contribution

Les contributions sont les bienvenues ! Si vous souhaitez améliorer ce projet, veuillez suivre les étapes suivantes :

1. **Fork** le dépôt.
2. **Créer une branche** pour votre fonctionnalité ou correction de bug.
3. **Commiter** vos changements avec des messages clairs.
4. **Pousser** votre branche vers votre fork.
5. **Créer une Pull Request** décrivant vos changements.

## Licence

Ce projet est sous licence [MIT](LICENSE).

## Contact

Pour toute question ou suggestion, veuillez contacter :

**Abraham KOLOBOE**  
[GitHub](https://github.com/AbrahamKOLOBOE)  
[Email](mailto:abraham.koloboe@example.com)



Merci d'utiliser le **Dashboard Analytics des Ventes - Rossmann Stores** ! Nous espérons que cette application vous sera utile pour optimiser la gestion de vos stocks et améliorer vos prévisions de ventes.

