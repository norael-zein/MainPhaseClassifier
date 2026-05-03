"""
Pipeline for predicting CT contrast phase from one .nii.gz file.

"""
from pathlib import Path
import argparse
import joblib
import numpy as np
import pandas as pd
from feature_extractor import get_features

MODEL_DIR = Path(__file__).resolve().parent
MISSING_VALUE = -9999

def load_model_files(model_dir):
    """Loads the trained model, selected feature names, and label encoder from the model folder."""
    model = joblib.load(model_dir / "final_model.pkl")
    feature_names = joblib.load(model_dir / "features.pkl")
    label_encoder = joblib.load(model_dir / "label_encoder.pkl")
    return model, feature_names, label_encoder

def predict_phase(input_path, fast=False, totalseg_model_dir=None):
    """Extracts model features from one CT image and predicts its contrast phase."""
    model, feature_names, label_encoder = load_model_files(MODEL_DIR)
    features = get_features(input_data=input_path,fast=fast,model_dir=totalseg_model_dir)
    X = features.reindex(columns=feature_names, fill_value=MISSING_VALUE)
    X = X.replace([np.inf, -np.inf], MISSING_VALUE)
    X = X.fillna(MISSING_VALUE)

    print("\nFeature check:")
    print("Number of expected features:", len(feature_names))
    print("Number of extracted columns:", len(features.columns))
    print("Number sent to model:", len(X.columns))
    print("Missing sent as -9999:", int((X == -9999).sum(axis=1).iloc[0]))
    print("\nFeatures sent to model:")
    print(X.T)
    predicted_number = model.predict(X)
    predicted_phase = label_encoder.inverse_transform(predicted_number)[0]

    probabilities = None
    if hasattr(model, "predict_proba"):
        proba_labels = label_encoder.inverse_transform(model.classes_)
        probabilities = pd.Series(model.predict_proba(X)[0],index=proba_labels).sort_values(ascending=False)
    return predicted_phase, probabilities, X

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to a .nii.gz CT image")
    parser.add_argument("--fast", action="store_true", help="Use fast TotalSegmentator")
    parser.add_argument("--totalseg-model-dir",default=None,help="Optional path to a local TotalSegmentator model folder")
    parser.add_argument("--save-features",default=None,help="Optional CSV path where the extracted model features are saved")
    args = parser.parse_args()
    phase, probabilities, X = predict_phase(input_path=args.input_path,fast=args.fast,totalseg_model_dir=args.totalseg_model_dir)
    print(f"\nPredicted phase: {phase}")

    if probabilities is not None:
        print("\nProbabilities:")
        for phase_name, value in probabilities.items():
            print(f"{phase_name}: {value:.3f}")

    if args.save_features is not None:
        X.to_csv(args.save_features, index=False)
        print(f"\nSaved features to: {args.save_features}")


if __name__ == "__main__":
    main()
