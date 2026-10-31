# Prévision des ventes de magasins (Rossmann)

Projet de portfolio de prévision des ventes quotidiennes à partir du jeu de données
de la compétition **Rossmann Store Sales**. Le dépôt compare une forêt aléatoire
(Random Forest) et XGBoost avec un découpage chronologique, au lieu d'un split
aléatoire qui exposerait le modèle à des dates futures.

## Objectifs

- Nettoyer et joindre les ventes et les caractéristiques des magasins.
- Construire des variables calendaires et magasin, sans utiliser les clients
  du jour (information qui n'est pas disponible au moment de la prévision).
- Comparer Random Forest et XGBoost sur une période de test chronologiquement
  postérieure aux données d'entraînement.
- Évaluer les prédictions avec MAE et RMSE, dans l'unité de ventes du dataset.

## Données

Le dataset est issu de la compétition Kaggle
[Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales/data).
Il contient les ventes quotidiennes de 1 115 magasins de janvier 2013 à juillet
2015. Les règles de la compétition et les conditions d'utilisation des données
s'appliquent.

1. Connectez-vous à Kaggle et acceptez les règles de la compétition.
2. Téléchargez `train.csv` et `store.csv`.
3. Placez les deux fichiers dans `data/raw/`.

Les données brutes ne sont pas incluses dans ce dépôt. Le script s'arrête avec
un message explicite si les fichiers ne sont pas présents.

## Installation et exécution

Python 3.9 ou plus récent est recommandé.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.forecasting
```

Le script entraîne les deux modèles et produit :

- `reports/metrics.csv` : MAE, RMSE et temps d'entraînement ;
- `reports/predictions.csv` : valeurs réelles et prédites sur le test ;
- `reports/figures/monthly_sales.png` et `test_predictions.png` : graphiques.

Pour utiliser un autre emplacement de données :

```powershell
python -m src.forecasting --data-dir "C:\chemin\vers\les\csv"
```

## Méthodologie

- Les magasins fermés ne sont pas des observations de demande et sont retirés.
- Les valeurs manquantes de distance à la concurrence sont imputées par la
  médiane.
- Le nombre de clients n'est pas utilisé comme variable explicative : il est
  mesuré pendant la journée et risquerait de créer une fuite de données.
- Les ventes sont transformées avec `log1p` pendant l'apprentissage, puis les
  prédictions sont reconverties avant le calcul des métriques.
- Les 42 derniers jours constituent le test ; les dates antérieures servent à
  l'entraînement. Pour limiter le temps de calcul, au plus 100 000 lignes
  d'entraînement sont sélectionnées avec une graine fixe (42). Le test reste
  intégral et identique pour les deux modèles.
- Les hyperparamètres ne sont pas présentés comme optimisés par validation
  croisée.

## Résultats mesurés

Sur l'exécution locale avec 100 000 lignes d'entraînement sélectionnées et les
42 derniers jours en test, le Random Forest obtient une MAE de **1 377,04** et
un RMSE de **1 939,42** ; XGBoost obtient une MAE de **1 466,50** et un RMSE de
**1 991,87** (unité du dataset). Sur ce découpage et ces paramètres, le Random
Forest est meilleur selon les deux métriques. Ces scores sont un point de
départ, pas une garantie de performance en production ; ils peuvent varier si
les données ou les versions des bibliothèques changent.

## Tests

```powershell
python -m unittest discover -s tests -v
```
