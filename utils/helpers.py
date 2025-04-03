import os

import pandas as pd
from joblib import load
from sklearn.preprocessing import MinMaxScaler

def load_dataset(project_path, file_type, path):
    file_path = os.path.join(path, f'{project_path}_{file_type}.csv')
    dataset = pd.read_csv(file_path)
    return dataset[dataset['isTerraform'] == 0]

def preprocess_data(dataset, features, target):
    X = dataset[features]
    y = dataset[target]
    return X, y

def get_project_paths(project_full_name):
    return {
        'project_path': project_full_name.replace('/', '__'),
        'project_model': project_full_name.replace('/', '_'),
        'project_local_folder': project_full_name.replace('/', '\\')
    }


def load_data(project_path, dataset_type, path):
    return load_dataset(project_path, dataset_type, path)


def load_model(project_local_folder, project_model, algo, i):
    """
    Loads a pre-trained machine learning model from disk.

    Args:
        project_local_folder (str): Path to local project folder (unused here).
        project_model (str): Path or name of model file (unused here).
        algo (str): The algorithm used (e.g., 'RF').
        i (int): Iteration index (unused here).

    Returns:
        sklearn.BaseEstimator: Loaded model object.
    """
    model_path = f'./saved_models/{project_model}_{algo}__iter_{i}_.joblib'
    return load(model_path)


def read_features(project, model, iteration):
    """
    Reads the list of selected features used to train the model.

    Args:
        project (str): Path to project (unused here).
        model (str): Model identifier (unused here).
        iteration (int): Iteration index (unused here).

    Returns:
        list[str]: List of feature names used for training/prediction.
    """
    local_path = f'./features/feature_importances_{model}_iter_{iteration}.csv'
    return pd.read_csv(local_path)['Feature'].tolist()


def predict_on_instance(raw_instance_df, scaler, model, feature_order):
    """
    Makes a prediction on a single incoming instance using the trained model and scaler.

    Args:
        raw_instance_df (pd.DataFrame): Unscaled single-row input instance.
        scaler (MinMaxScaler): Fitted scaler used during training.
        model (sklearn.BaseEstimator): Trained classification model.
        feature_order (list[str]): Ordered list of features expected by the model.

    Returns:
        tuple: (predicted_label: int, predicted_probability: float)
    """
    # Ensure the input features are in the expected order
    raw_instance_df = raw_instance_df[feature_order]

    # Scale the features using the same scaler used in training
    scaled_instance = scaler.transform(raw_instance_df.values)

    # Perform prediction
    y_prob = model.predict_proba(scaled_instance)[0, 1]
    y_pred = int(y_prob >= 0.5)

    return y_pred, y_prob


def load_scaler(model_name):
    """
    Loads a previously saved MinMaxScaler from disk.

    Returns:
        MinMaxScaler: The trained and saved scaler.
    """
    return load(f'./saved_models/scaler_{model_name}.joblib')