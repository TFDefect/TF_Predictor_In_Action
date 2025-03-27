
# 🧠 Prédicteur de Défectuosité pour les Projets Terraform

Ce projet montre comment utiliser un modèle de machine learning pré-entraîné pour **prédire si une modification de code Infrastructure-as-Code (IaC), en particulier du code Terraform, est potentiellement défectueuse**.

Le système traite les instances en temps réel ou en lot, et classe chaque instance comme :

- ✅ **CLEAN** – si elle est prédite comme non-défectueuse
- 🐞 **BUG** – si elle est prédite comme potentiellement fautive

---

## 📦 Structure du Projet

### 🗂️ Fichiers principaux

| Fichier | Description |
|--------|-------------|
| `main.py` | Point d’entrée principal pour tester une prédiction sur une instance de test |
| `utils/helpers.py` | Fonctions utilitaires pour charger les données, le modèle, les caractéristiques, et effectuer une prédiction |
| `saved_models/` | Contient le modèle entraîné (`.joblib`) et le scaler MinMax |
| `features/` | Contient la liste des caractéristiques sélectionnées |
| `historical_data/` | Contient les jeux de données d’apprentissage et de test au format CSV |

---

## 🚀 Mise en Route

### 1. Préparer les fichiers nécessaires

Assurez-vous que les fichiers suivants sont présents aux bons endroits :
- Modèle entraîné : `./saved_models/cattle-ops_terraform-aws-gitlab-runner_RF__iter_0_.joblib`
- Scaler MinMax entraîné : `./saved_models/scaler_RF.joblib`
- Liste des caractéristiques : `./features/feature_importances_RF_iter_0.csv`
- Jeu de test : `./historical_data/cattle-ops__terraform-aws-gitlab-runner_test.csv`

> ✅ Remarque : Le nom du projet suit le format GitHub (`organisation/repo`) qui est automatiquement adapté pour un usage local.

---

## 🧩 Fonctions et Composants

### `main.py`

Ce script :
- Charge les données de test pour un projet donné
- Charge un modèle et un scaler pré-entraînés
- Charge les caractéristiques sélectionnées
- Prédit si une instance donnée (ex : instance 18 dans le tableau test) est fautive
- Affiche la probabilité, l’étiquette prédite, l’étiquette réelle, et un icône 🐞 ou ✅

### `helpers.py`

#### `get_project_paths(project_full_name)`
Convertit le nom du projet GitHub (`organisation/repo`) en chemins compatibles pour les fichiers locaux.

#### `load_dataset(project_path, file_type, path)`
Lit un jeu de données CSV (train/test).

#### `load_data(project_path, dataset_type, path)`
Appel standard à `load_dataset`.

#### `load_model(...)`
Charge un modèle entraîné depuis le disque.

#### `read_features(...)`
Charge la liste des caractéristiques utilisées lors de l’entraînement du modèle.

#### `load_scaler()`
Charge un scaler `MinMaxScaler` sauvegardé pendant l’entraînement.

#### `predict_on_instance(raw_instance_df, scaler, model, feature_order)`
Fonction principale de prédiction :
- Réordonne les caractéristiques
- Applique le scaling
- Fait la prédiction
- Retourne l’étiquette binaire et la probabilité

---

## 🧪 Exemple de Résultat

```
Predicted probability: 0.7832
Predicted label: 1
Actual label: 1
==> 🐞 BUG
```

---

## 🛠️ Personnalisation

- 🔁 **Changer d’instance à tester** : Modifier `instance_index` dans `main.py`
- 🧪 **Tester plusieurs instances** : Boucler sur le jeu de test
- 🧠 **Réentraîner le modèle** : Générer un nouveau modèle, scaler et liste de caractéristiques

---

## 📋 Prérequis

- Python 3.7+
- scikit-learn
- pandas
- joblib

Installer les dépendances :

```bash
pip install -r requirements.txt
```

> Attention : utilisez la même version de scikit-learn que celle utilisée lors de l’entraînement du modèle (par ex. `scikit-learn==1.3.1`) pour éviter les avertissements.

---

## 🔐 Remarques

- Le modèle et le scaler doivent avoir été entraînés avec les **mêmes caractéristiques** dans **le même ordre**.
Le modèle ici s’applique uniquement aux commits non liés aux blocs Terraform spécifiquement associés au versionnement (isTerraform == 0). Vous pouvez adapter ce filtrage dans load_dataset().
---

## 🔮 Améliorations Futures

- [ ] Déployer une API REST pour des prédictions en temps réel
- [ ] Créer un tableau de bord visuel pour surveiller les prédictions
- [ ] Automatiser l’extraction de caractéristiques à partir de commits bruts

---

## 👩‍💻 Auteur

*Cet outil s’inscrit dans un projet de recherche sur la dette technique et la qualité du code dans les scripts Infrastructure-as-Code.*
