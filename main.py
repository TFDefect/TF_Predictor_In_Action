import warnings

from sklearn.exceptions import InconsistentVersionWarning

from utils.helpers import (
    get_project_paths,
    load_data, load_model, load_scaler, read_features, predict_on_instance,
)

# Suppress version mismatch warnings when loading sklearn models
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Path to historical train/test split data
path = './historical_data/'


if __name__ == '__main__':
    # --- Configuration ---
    clf_name = 'RF'
    project_full_name = 'cattle-ops/terraform-aws-gitlab-runner'

    # --- Load paths and test set for simulation ---
    paths = get_project_paths(project_full_name)
    test_set = load_data(paths['project_path'], 'test', path)

    # --- Load pretrained model and scaler ---
    model = load_model(paths['project_local_folder'], paths['project_model'], clf_name, 0)
    scaler = load_scaler()

    # --- Load selected features and define feature order ---
    non_correlated_features = read_features(paths['project_local_folder'], clf_name, 0)
    feature_order = non_correlated_features  # Assumes order is preserved in CSV

    # --- Simulate prediction on a single incoming instance ---
    instance_index = 18  # Change as needed for testing different rows
    raw_instance_df = test_set[non_correlated_features].iloc[[instance_index]]
    actual_label = test_set['fault_prone'].iloc[instance_index]

    predicted_label, predicted_prob = predict_on_instance(raw_instance_df, scaler, model, feature_order)

    # --- Output results ---
    print(f"Predicted probability: {predicted_prob:.4f}")
    print(f"Predicted label: {predicted_label}")
    print(f"Actual label: {actual_label}")

    # --- Print BUG or CLEAN based on prediction with icons ---
    if predicted_prob > 0.5:
        print("==> 🐞 BUG")
    else:
        print("==> ✅ CLEAN")
