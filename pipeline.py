"""
Use this script to run the entire pipeline.

Run:
python pipeline.py --input "input_path" --output "output_path"
"""

import os
import argparse
import numpy as np
import joblib
import pandas as pd
import SimpleITK as sitk

from run_totalsegmentator import segment_selected_organs
from feature_extractor import extract_features, calculate_feature_differences
from totalsegmentator.map_to_binary import class_map


ORGANS = [
    "aorta",
    "portal_vein_and_splenic_vein",
    "urinary_bladder",
    "kidney_right",
    "kidney_left",
    "spleen",
    "liver",
]

FEATURE_KEYS = {
    "Mean": "original_firstorder_Mean",
    "Standard Deviation": "original_firstorder_StandardDeviation",
    "10th Percentile": "original_firstorder_10Percentile",
    "50th Percentile": "original_firstorder_Median",
    "90th Percentile": "original_firstorder_90Percentile",
    "Median": "original_firstorder_Median",
    "Skewness": "original_firstorder_Skewness",
    "Kurtosis": "original_firstorder_Kurtosis",
    "Entropy": "original_firstorder_Entropy",
    "Uniformity": "original_firstorder_Uniformity",
    "Variance": "original_firstorder_Variance",
    "InterquartileRange (IQR)": "original_firstorder_InterquartileRange",
    "GLCM_Contrast": "original_glcm_Contrast",
    "GLCM_Correlation": "original_glcm_Correlation",
    "GLCM_JointEntropy": "original_glcm_JointEntropy",
    "GLSZM_ZoneVariance": "original_glszm_ZoneVariance",
}


def read_ct_image(input_data):
    if os.path.isdir(input_data):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(input_data)

        if len(dicom_names) == 0:
            raise ValueError(f"No DICOM files found in folder: {input_data}")

        reader.SetFileNames(dicom_names)
        return reader.Execute()

    return sitk.ReadImage(input_data)


def combine_kidneys(right_features, left_features):
    kidneys = {}
    all_keys = set(right_features.keys()) | set(left_features.keys())

    for key in all_keys:
        values = []

        for features in [right_features, left_features]:
            if key in features:
                try:
                    values.append(float(features[key]))
                except (TypeError, ValueError):
                    pass

        if values:
            kidneys[key] = float(np.mean(values))

    return kidneys


def add_feature(final, organ_features, output_name, feature_name, organ_name):
    key = FEATURE_KEYS[feature_name]

    if organ_name in organ_features and key in organ_features[organ_name]:
        final[output_name] = float(organ_features[organ_name][key])
    else:
        final[output_name] = None


def get_diff(differences, feature_name, organ_a, organ_b):
    key = FEATURE_KEYS[feature_name]
    diff_group = differences.get(f"{key}_differences", {})

    direct_key = f"{organ_a}_vs_{organ_b}"
    reverse_key = f"{organ_b}_vs_{organ_a}"

    if direct_key in diff_group:
        return float(diff_group[direct_key])

    if reverse_key in diff_group:
        return -float(diff_group[reverse_key])

    return None


def add_diff(final, differences, output_name, feature_name, organ_a, organ_b):
    final[output_name] = get_diff(differences, feature_name, organ_a, organ_b)


def nibabel_mask_to_sitk(mask_np, reference_sitk):
    mask_sitk = sitk.GetImageFromArray(mask_np.transpose(2, 1, 0).astype(np.uint8))
    mask_sitk.CopyInformation(reference_sitk)
    return mask_sitk


def pipeline(input_data, output_path):
    os.makedirs(output_path, exist_ok=True)

    seg_img = segment_selected_organs(
        input_data=input_data,
        organs=ORGANS,
    )

    original_image = read_ct_image(input_data)

    image_np = sitk.GetArrayFromImage(original_image)

    # Clip HU values to match training preprocessing
    image_np = np.clip(image_np, -200, 200)

    image_for_features = sitk.GetImageFromArray(image_np)
    image_for_features.CopyInformation(original_image)

    seg_data = np.asanyarray(seg_img.dataobj)

    task_class_map = class_map["total"]
    organ_to_label = {
        organ_name: label_id
        for label_id, organ_name in task_class_map.items()
    }

    selected_features = list(set(FEATURE_KEYS.values()))
    organ_features = {}

    for organ in ORGANS:
        if organ not in organ_to_label:
            print(f"Varning: {organ} finns inte i class_map")
            continue

        label_id = organ_to_label[organ]
        mask_np = seg_data == label_id

        if mask_np.sum() == 0:
            print(f"Varning: tom mask för {organ}")
            continue

        mask_sitk = nibabel_mask_to_sitk(mask_np, image_for_features)

        organ_features[organ] = extract_features(
            image_input=image_for_features,
            mask_input=mask_sitk,
            selected_features=selected_features,
        )

    if "kidney_right" in organ_features and "kidney_left" in organ_features:
        organ_features["kidneys"] = combine_kidneys(
            organ_features["kidney_right"],
            organ_features["kidney_left"],
        )

    differences = calculate_feature_differences(organ_features)
    final = {}

    for organ in [
        "aorta",
        "kidneys",
        "liver",
        "portal_vein_and_splenic_vein",
        "spleen",
        "urinary_bladder",
    ]:
        add_feature(final, organ_features, f"Mean_{organ}", "Mean", organ)
        add_feature(final, organ_features, f"Standard Deviation_{organ}", "Standard Deviation", organ)
        add_feature(final, organ_features, f"10th Percentile_{organ}", "10th Percentile", organ)
        add_feature(final, organ_features, f"50th Percentile_{organ}", "50th Percentile", organ)
        add_feature(final, organ_features, f"90th Percentile_{organ}", "90th Percentile", organ)

    add_feature(final, organ_features, "Skewness_spleen", "Skewness", "spleen")
    add_feature(final, organ_features, "Kurtosis_spleen", "Kurtosis", "spleen")
    add_feature(final, organ_features, "Entropy_spleen", "Entropy", "spleen")
    add_feature(final, organ_features, "Uniformity_spleen", "Uniformity", "spleen")

    add_feature(final, organ_features, "Skewness_liver", "Skewness", "liver")
    add_feature(final, organ_features, "Kurtosis_liver", "Kurtosis", "liver")
    add_feature(final, organ_features, "Entropy_liver", "Entropy", "liver")
    add_feature(final, organ_features, "Uniformity_liver", "Uniformity", "liver")
    add_feature(final, organ_features, "GLCM_Contrast_liver", "GLCM_Contrast", "liver")
    add_feature(final, organ_features, "GLCM_Correlation_liver", "GLCM_Correlation", "liver")
    add_feature(final, organ_features, "GLCM_JointEntropy_liver", "GLCM_JointEntropy", "liver")
    add_feature(final, organ_features, "InterquartileRange (IQR)_liver", "InterquartileRange (IQR)", "liver")
    add_feature(final, organ_features, "Variance_liver", "Variance", "liver")

    add_feature(final, organ_features, "Kurtosis_kidneys", "Kurtosis", "kidneys")
    add_feature(final, organ_features, "Skewness_kidneys", "Skewness", "kidneys")

    add_feature(final, organ_features, "GLSZM_ZoneVariance_urinary_bladder", "GLSZM_ZoneVariance", "urinary_bladder")

    add_feature(final, organ_features, "Entropy_portal_vein_and_splenic_vein", "Entropy", "portal_vein_and_splenic_vein")
    add_feature(final, organ_features, "Uniformity_portal_vein_and_splenic_vein", "Uniformity", "portal_vein_and_splenic_vein")

    add_diff(final, differences, "Mean_diff_aorta_portal_vein_and_splenic_vein", "Mean", "aorta", "portal_vein_and_splenic_vein")
    add_diff(final, differences, "Median_diff_aorta_liver", "Median", "aorta", "liver")
    add_diff(final, differences, "Median_diff_kidneys_portal_vein_and_splenic_vein", "Median", "kidneys", "portal_vein_and_splenic_vein")
    add_diff(final, differences, "Median_diff_aorta_portal_vein_and_splenic_vein", "Median", "aorta", "portal_vein_and_splenic_vein")
    add_diff(final, differences, "Mean_diff_liver_portal_vein_and_splenic_vein", "Mean", "liver", "portal_vein_and_splenic_vein")
    add_diff(final, differences, "Median_diff_liver_portal_vein_and_splenic_vein", "Median", "liver", "portal_vein_and_splenic_vein")
    add_diff(final, differences, "Median_diff_aorta_kidneys", "Median", "aorta", "kidneys")
    add_diff(final, differences, "Mean_diff_aorta_liver", "Mean", "aorta", "liver")
    add_diff(final, differences, "10th Percentile_diff_aorta_kidneys", "10th Percentile", "aorta", "kidneys")
    add_diff(final, differences, "50th Percentile_diff_aorta_kidneys", "50th Percentile", "aorta", "kidneys")
    add_diff(final, differences, "Mean_diff_liver_spleen", "Mean", "liver", "spleen")
    add_diff(final, differences, "Mean_diff_kidneys_portal_vein_and_splenic_vein", "Mean", "kidneys", "portal_vein_and_splenic_vein")

    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CTPhaseClassification Pipeline")

    parser.add_argument("--input", required=True, help="Path to input CT NIfTI file or DICOM folder")
    parser.add_argument("--output", required=True, help="Output folder")
    parser.add_argument("--model", default="final_model.pkl")
    parser.add_argument("--features", default="features.pkl")
    parser.add_argument("--label_encoder", default="label_encoder.pkl")

    args = parser.parse_args()

    features = pipeline(args.input, args.output)

    print("\n=== FINAL FEATURES ===")
    for key, value in features.items():
        print(f"{key}: {value}")

    model = joblib.load(args.model)
    selected_feature_names = joblib.load(args.features)
    label_encoder = joblib.load(args.label_encoder)

    X_new = pd.DataFrame([features])

    missing_features = []

    for feature in selected_feature_names:
        if feature not in X_new.columns:
            X_new[feature] = -9999
            missing_features.append(feature)

    if missing_features:
        print("\n=== WARNING: MISSING FEATURES FILLED WITH -9999 ===")
        for feature in missing_features:
            print(feature)

    X_new = X_new.replace({None: np.nan})
    X_new = X_new.apply(pd.to_numeric, errors="coerce")
    X_new = X_new.fillna(-9999)
    X_new = X_new[selected_feature_names]

    y_pred = model.predict(X_new)
    y_label = label_encoder.inverse_transform(y_pred)

    print("\n=== PREDICTION ===")
    print("Predicted class:", y_label[0])