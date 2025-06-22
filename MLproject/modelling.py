import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import random
import numpy as np
from joblib import load
import os
import warnings
import sys

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    X_train = pd.read_csv(os.path.join(script_dir, 'dataset_preprocessing', 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(script_dir, 'dataset_preprocessing', 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(script_dir, 'dataset_preprocessing', 'y_train.csv'))
    y_train = y_train.to_numpy().ravel()  # Convert to
    y_test = pd.read_csv(os.path.join(script_dir, 'dataset_preprocessing', 'y_test.csv'))
    y_test = y_test.to_numpy().ravel()  # Convert to 1D array
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 505
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 37

    def inference(new_data, load_path):
        # Memuat pipeline preprocessing
        preprocessor = load(load_path)

        # Transformasi data baru
        transformed_data = preprocessor.transform(new_data)
        return transformed_data

    def inverse_transform_data(transformed_data, load_path):
        preprocessor = load(load_path)

        # === Extract transformers correctly ===
        _, int_transformer, int_cols = preprocessor.transformers_[0]
        _, float_transformer, float_cols = preprocessor.transformers_[1]
        _, cat_transformer, cat_cols = preprocessor.transformers_[2]

        # === Get the internal scalers and encoder ===
        int_scaler = int_transformer.named_steps['scaler']
        float_scaler = float_transformer.named_steps['scaler']
        ohe = cat_transformer.named_steps['encoder']

        # === Split the transformed data into each block ===
        int_end = len(int_cols)
        float_end = int_end + len(float_cols)

        int_data = transformed_data[:, :int_end]
        float_data = transformed_data[:, int_end:float_end]
        cat_data = transformed_data[:, float_end:]

        # === Inverse transforms ===
        int_orig = int_scaler.inverse_transform(int_data)
        float_orig = float_scaler.inverse_transform(float_data)
        cat_orig = ohe.inverse_transform(cat_data)

        # === Rebuild DataFrame ===
        df_int = pd.DataFrame(int_orig, columns=int_cols)
        df_float = pd.DataFrame(float_orig, columns=float_cols)
        df_cat = pd.DataFrame(cat_orig, columns=cat_cols)

        # Combine all parts
        df_reconstructed = pd.concat([df_int, df_float, df_cat], axis=1)
        return df_reconstructed

    # preprocess_pipeline_path = os.path.join(script_dir, 'dataset_preprocessing', 'preprocessor_pipeline.joblib')
    # preprocessor = load(preprocess_pipeline_path)
    # inversed_data = inverse_transform_data(X_train, preprocess_pipeline_path)

    # input_example = inversed_data.sample(n=5, random_state=random.randint(0, 1000))
    with mlflow.start_run():
        mlflow.autolog()
        # Train model
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_train[:5],  # Use a small sample for input example
        )
        # Log metrics
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)