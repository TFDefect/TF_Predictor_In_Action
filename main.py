import warnings
import os
import pandas as pd
from joblib import load, dump
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.preprocessing import MinMaxScaler

from utils.helpers import (
    get_project_paths,
    load_data, load_model, read_features, predict_on_instance, preprocess_data,
)

# Suppress version mismatch warnings when loading sklearn models
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Path to historical train/test split data
path = './historical_data/'

if __name__ == '__main__':
    # --- Models to Evaluate ---
    models_mapping = [
        {"clf_name": "RF", "project_full_name": "cattle-ops/terraform-aws-gitlab-runner"},
        {"clf_name": "LR", "project_full_name": "cookpad/terraform-aws-eks"},
        {"clf_name": "LightGBM", "project_full_name": "CDCgov/prime-simplereport"},
        {"clf_name": "NB", "project_full_name": "oracle-terraform-modules/terraform-oci-oke"},
    ]

    instance_index = 18  # You can vary this per model if needed

    for entry in models_mapping:
        clf_name = entry['clf_name']
        project_full_name = entry['project_full_name']

        print(f"\n=== Evaluating: {clf_name} on {project_full_name} ===")

        # --- Load paths and test set for simulation ---
        paths = get_project_paths(project_full_name)
        test_set = load_data(paths['project_path'], 'test', path)

        # --- Load pretrained model ---
        model = load_model(paths['project_local_folder'], paths['project_model'], clf_name, 0)

        # --- Load selected features and define feature order ---
        non_correlated_features = read_features(paths['project_local_folder'], clf_name, 0)
        feature_order = non_correlated_features

        # --- Load or build scaler ---
        scaler_path = f"./saved_models/scaler_{paths['project_model']}_{clf_name}.joblib"
        if os.path.exists(scaler_path):
            scaler = load(scaler_path)
            print("Loaded existing scaler.")
        else:
            print("Scaler not found. Building new one...")
            train_set = load_data(paths['project_path'], 'train', path)
            X_train, y_train = preprocess_data(train_set, non_correlated_features, 'fault_prone')
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train.values)
            dump(scaler, scaler_path)
            print(f"Scaler saved to: {scaler_path}")



        # --- Simulate prediction on a single instance ---
        raw_instance_df = test_set[non_correlated_features].iloc[[instance_index]]
        actual_label = test_set['fault_prone'].iloc[instance_index]

        predicted_label, predicted_prob = predict_on_instance(raw_instance_df, scaler, model, feature_order)

        # --- Output results ---
        print(f"Predicted probability: {predicted_prob:.4f}")
        print(f"Predicted label: {predicted_label}")
        print(f"Actual label: {actual_label}")
        print("==> 🐞 BUG" if predicted_prob > 0.5 else "==> ✅ CLEAN")
