"""
Functions for extracting radiomic features from CT images and their corresponding masks. 
"""
import os
import glob
import pandas as pd
import numpy as np
import SimpleITK as sitk
import tempfile
from radiomics import featureextractor

def extract_features(image_path, mask_path):
    """
    Extract radiomic features from a given image and a mas using PyRadiomics.
    """
    #<imageType>_<featureClass>_<featureName> in the feature names
    extractor = featureextractor.RadiomicsFeatureExtractor()
    features = extractor.execute(image_path, mask_path)
    return features


def selected_features(features):
    """
    Select a subset of features to use for feature extraction.
    """
    selected_features = {
        "Mean": float(features["original_firstorder_Mean"]),
        "Median": float(features["original_firstorder_Median"]),
        "Variance": float(features["original_firstorder_Variance"]),
        "Standard Deviation": float(np.sqrt(float(features["original_firstorder_Variance"]))),
        "10th Percentile": float(features["original_firstorder_10Percentile"]),
        "50th Percentile": float(features["original_firstorder_Median"]),
        "90th Percentile": float(features["original_firstorder_90Percentile"]),
        "InterquartileRange (IQR)": float(features["original_firstorder_InterquartileRange"]),
        "RootMeanSquared (RMS)": float(features["original_firstorder_RootMeanSquared"]),
        "Skewness": float(features["original_firstorder_Skewness"]),
        "Kurtosis": float(features["original_firstorder_Kurtosis"]),
        "Entropy": float(features["original_firstorder_Entropy"]),
        "Uniformity": float(features["original_firstorder_Uniformity"]),
        "GLCM_Contrast": float(features["original_glcm_Contrast"]),
        "GLCM_Correlation": float(features["original_glcm_Correlation"]),
        "GLCM_JointEntropy": float(features["original_glcm_JointEntropy"]),
        "GLRLM_RunEntropy": float(features["original_glrlm_RunEntropy"]),
        "GLRLM_RunVariance": float(features["original_glrlm_RunVariance"]),
        "GLSZM_ZoneEntropy": float(features["original_glszm_ZoneEntropy"]),
        "GLSZM_ZoneVariance": float(features["original_glszm_ZoneVariance"]),
        "GLDM_DependenceEntropy": float(features["original_gldm_DependenceEntropy"]),
        "GLDM_DependenceVariance": float(features["original_gldm_DependenceVariance"]),
    }
    return selected_features

import glob  # <-- lägg till

def get_features_for_organ(image_path, mask_dir, organ_name, mask_spec):
    """
    Get radiomic features for a specific organ.
    Special cases:
      - kidneys: combine mask_spec[0] and mask_spec[1]
      - muscles: combine all masks in abdominal_muscles/ (and optionally tissue_types/skeletal_muscle.nii.gz)
      - fat: combine tissue_types/subcutaneous_fat.nii.gz + tissue_types/torso_fat.nii.gz
    """

    if organ_name == "kidneys":
        path_r = os.path.join(mask_dir, mask_spec[0])
        path_l = os.path.join(mask_dir, mask_spec[1])

        if not (os.path.exists(path_r) and os.path.exists(path_l)):
            return None

        mask_r = sitk.ReadImage(path_r)
        mask_l = sitk.ReadImage(path_l)
        combined = sitk.Cast(sitk.Or(mask_r > 0, mask_l > 0), sitk.sitkUInt8)

        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
            tmp_path = tmp.name

        sitk.WriteImage(combined, tmp_path)

        try:
            features = extract_features(image_path, tmp_path)
        except ValueError:
            os.remove(tmp_path)
            return None

        os.remove(tmp_path)
        return features

    elif organ_name == "muscles":
        abd_dir = os.path.join(mask_dir, "abdominal_muscles")
        tissue_muscle = os.path.join(mask_dir, "tissue_types", "skeletal_muscle.nii.gz")

        mask_paths = []
        if os.path.isdir(abd_dir):
            mask_paths.extend(glob.glob(os.path.join(abd_dir, "*.nii.gz")))
        if os.path.exists(tissue_muscle):
            mask_paths.append(tissue_muscle)

        if len(mask_paths) == 0:
            return None

        combined = None
        for p in mask_paths:
            m = sitk.ReadImage(p)
            m = sitk.Cast(m > 0, sitk.sitkUInt8)
            combined = m if combined is None else sitk.Or(combined > 0, m > 0)
        combined = sitk.Cast(combined > 0, sitk.sitkUInt8)

        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
            tmp_path = tmp.name
        sitk.WriteImage(combined, tmp_path)

        try:
            features = extract_features(image_path, tmp_path)
        except ValueError:
            os.remove(tmp_path)
            return None

        os.remove(tmp_path)
        return features

    elif organ_name == "fat":
        subq = os.path.join(mask_dir, "tissue_types", "subcutaneous_fat.nii.gz")
        torso = os.path.join(mask_dir, "tissue_types", "torso_fat.nii.gz")

        paths = [p for p in [subq, torso] if os.path.exists(p)]
        if len(paths) == 0:
            return None

        combined = None
        for p in paths:
            m = sitk.ReadImage(p)
            m = sitk.Cast(m > 0, sitk.sitkUInt8)
            combined = m if combined is None else sitk.Or(combined > 0, m > 0)
        combined = sitk.Cast(combined > 0, sitk.sitkUInt8)

        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
            tmp_path = tmp.name
        sitk.WriteImage(combined, tmp_path)

        try:
            features = extract_features(image_path, tmp_path)
        except ValueError:
            os.remove(tmp_path)
            return None

        os.remove(tmp_path)
        return features

    else:
        mask_path = os.path.join(mask_dir, mask_spec)
        if not os.path.exists(mask_path):
            return None

        try:
            return extract_features(image_path, mask_path)
        except ValueError:
            return None


def calculate_organ_difference(df, organ1, organ2, features_to_compare, missing_value=-9999):
    """
    Calculate difference (organ1-organ2) for selected features within same case.
    If either value is missing_value, set diff to missing_value.
    """
    diff_rows = []

    for case in df["case"].unique():
        df_case = df[df["case"] == case]

        row1 = df_case[df_case["organ"] == organ1]
        row2 = df_case[df_case["organ"] == organ2]

        if row1.empty or row2.empty:
            continue

        row1 = row1.iloc[0]
        row2 = row2.iloc[0]

        diff = {"case": case, "organ_pair": f"{organ1}-{organ2}"}

        for feat in features_to_compare:
            v1 = row1[feat]
            v2 = row2[feat]

            if v1 == missing_value or v2 == missing_value:
                diff[f"{feat}_diff"] = missing_value
            else:
                diff[f"{feat}_diff"] = v1 - v2

        diff_rows.append(diff)

    return pd.DataFrame(diff_rows)