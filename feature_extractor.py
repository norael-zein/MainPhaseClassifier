"""
Feature extraction used for CT phase prediction.
"""
import os
import tempfile
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from totalsegmentator.map_to_binary import class_map
from run_totalsegmentator import segment_selected_organs


#Handle missing values = -9999
MISSING_VALUE = -9999


_EXTRACTOR = None
ORGANS = ["aorta","portal_vein_and_splenic_vein","urinary_bladder","kidney_right","kidney_left","spleen","liver"]

#Feature input
FINAL_FEATURES = [
    "Mean_aorta",
    "Mean_kidneys",
    "Mean_liver",
    "Mean_portal_vein_and_splenic_vein",
    "Mean_spleen",
    "Mean_urinary_bladder",
    "Standard Deviation_aorta",
    "Standard Deviation_kidneys",
    "Standard Deviation_liver",
    "Standard Deviation_portal_vein_and_splenic_vein",
    "Standard Deviation_spleen",
    "Standard Deviation_urinary_bladder",
    "10th Percentile_aorta",
    "10th Percentile_kidneys",
    "10th Percentile_liver",
    "10th Percentile_portal_vein_and_splenic_vein",
    "10th Percentile_spleen",
    "10th Percentile_urinary_bladder",
    "50th Percentile_aorta",
    "50th Percentile_kidneys",
    "50th Percentile_liver",
    "50th Percentile_portal_vein_and_splenic_vein",
    "50th Percentile_spleen",
    "50th Percentile_urinary_bladder",
    "90th Percentile_aorta",
    "90th Percentile_kidneys",
    "90th Percentile_liver",
    "90th Percentile_portal_vein_and_splenic_vein",
    "90th Percentile_spleen",
    "90th Percentile_urinary_bladder",
    "Skewness_spleen",
    "Kurtosis_spleen",
    "Entropy_spleen",
    "Uniformity_spleen",
    "Mean_diff_aorta_portal_vein_and_splenic_vein",
    "Median_diff_aorta_liver",
    "Median_diff_kidneys_portal_vein_and_splenic_vein",
    "Median_diff_aorta_portal_vein_and_splenic_vein",
    "Mean_diff_liver_portal_vein_and_splenic_vein",
    "Median_diff_liver_portal_vein_and_splenic_vein",
    "Skewness_liver",
    "Kurtosis_liver",
    "Entropy_liver",
    "Uniformity_liver",
    "GLCM_Contrast_liver",
    "GLCM_Correlation_liver",
    "GLCM_JointEntropy_liver",
    "InterquartileRange (IQR)_liver",
    "Variance_liver",
    "Kurtosis_kidneys",
    "Median_diff_aorta_kidneys",
    "Skewness_kidneys",
    "Mean_diff_aorta_liver",
    "10th Percentile_diff_aorta_kidneys",
    "50th Percentile_diff_aorta_kidneys",
    "GLSZM_ZoneVariance_urinary_bladder",
    "Entropy_portal_vein_and_splenic_vein",
    "Uniformity_portal_vein_and_splenic_vein",
    "Mean_diff_liver_spleen",
    "Mean_diff_kidneys_portal_vein_and_splenic_vein",
]

def get_extractor():
    global _EXTRACTOR
    if _EXTRACTOR is None:
        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.disableAllFeatures()
        extractor.enableFeaturesByName(
            firstorder=["Mean","Median","Variance","10Percentile","90Percentile","InterquartileRange","RootMeanSquared","Skewness","Kurtosis","Entropy","Uniformity"],
            glcm=["Contrast", "Correlation", "JointEntropy"],
            glszm=["ZoneVariance"],
        )
        extractor.disableAllImageTypes()
        extractor.enableImageTypeByName("Original")
        _EXTRACTOR = extractor
    return _EXTRACTOR

def read_ct_image(input_data):
    """Reads the CT image from a SimpleITK image, DICOM folder, or image file path."""
    if isinstance(input_data, sitk.Image):
        return input_data
    if os.path.isdir(str(input_data)):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(input_data))
        if len(dicom_names) == 0:
            raise ValueError(f"No DICOM files found in folder: {input_data}")
        reader.SetFileNames(dicom_names)
        return reader.Execute()
    return sitk.ReadImage(str(input_data))

def clip_image(image):
    """Clips CT intensity values to the range [-200, 200]."""
    image_np = sitk.GetArrayFromImage(image)
    image_np = np.clip(image_np, -200, 200)
    clipped = sitk.GetImageFromArray(image_np)
    clipped.CopyInformation(image)
    return clipped

def nibabel_mask_to_sitk(mask_np, reference_sitk):
    """Converts a segmentation mask from nibabel/numpy format to SimpleITK format using the CT image as reference."""    
    mask_sitk = sitk.GetImageFromArray(mask_np.transpose(2, 1, 0).astype(np.uint8))
    mask_sitk.CopyInformation(reference_sitk)
    return mask_sitk

def clean_value(value):
    """Converts invalid numerical values, such as NaN or infinity, to the predefined missing value = -9999."""
    value = float(value)
    if np.isnan(value) or np.isinf(value):
        return MISSING_VALUE
    return value


def selected_features(raw_features, label_name):
    """Selects and renames the radiomic features that are relevant for one segmented organ."""
    variance = clean_value(raw_features["original_firstorder_Variance"])
    return {
        f"Mean_{label_name}": clean_value(raw_features["original_firstorder_Mean"]),
        f"Median_{label_name}": clean_value(raw_features["original_firstorder_Median"]),
        f"Variance_{label_name}": variance,
        f"Standard Deviation_{label_name}": clean_value(np.sqrt(variance)),
        f"10th Percentile_{label_name}": clean_value(raw_features["original_firstorder_10Percentile"]),
        f"50th Percentile_{label_name}": clean_value(raw_features["original_firstorder_Median"]),
        f"90th Percentile_{label_name}": clean_value(raw_features["original_firstorder_90Percentile"]),
        f"InterquartileRange (IQR)_{label_name}": clean_value(raw_features["original_firstorder_InterquartileRange"]),
        f"RootMeanSquared (RMS)_{label_name}": clean_value(raw_features["original_firstorder_RootMeanSquared"]),
        f"Skewness_{label_name}": clean_value(raw_features["original_firstorder_Skewness"]),
        f"Kurtosis_{label_name}": clean_value(raw_features["original_firstorder_Kurtosis"]),
        f"Entropy_{label_name}": clean_value(raw_features["original_firstorder_Entropy"]),
        f"Uniformity_{label_name}": clean_value(raw_features["original_firstorder_Uniformity"]),
        f"GLCM_Contrast_{label_name}": clean_value(raw_features["original_glcm_Contrast"]),
        f"GLCM_Correlation_{label_name}": clean_value(raw_features["original_glcm_Correlation"]),
        f"GLCM_JointEntropy_{label_name}": clean_value(raw_features["original_glcm_JointEntropy"]),
        f"GLSZM_ZoneVariance_{label_name}": clean_value(raw_features["original_glszm_ZoneVariance"]),
    }

def mask_is_empty(mask):
    """Checks whether a segmentation mask contains any voxels."""
    return np.sum(sitk.GetArrayFromImage(mask) > 0) == 0

def print_mask_intensity_check(image, mask, label_name):
    """Prints basic intensity statistics for an organ mask as a quality check before feature extraction."""
    image_arr = sitk.GetArrayFromImage(image)
    mask_arr = sitk.GetArrayFromImage(mask) > 0
    if not np.any(mask_arr):
        print(f"{label_name}: empty mask")
        return
    values = image_arr[mask_arr]
    clipped_low_percent = np.mean(values <= -199) * 100
    print(f"{label_name}: voxels={values.size}, "f"mean={np.mean(values):.2f}, median={np.median(values):.2f}, "f"<=-199={clipped_low_percent:.2f}%")

def extract_features_for_mask(image, mask, label_name):
    """Extracts radiomic features from one organ mask and handles empty masks or extraction errors."""
    if mask_is_empty(mask):
        print(f"Warning: empty mask for {label_name}")
        return {}
    print_mask_intensity_check(image, mask, label_name)
    extractor = get_extractor()
    try:
        raw_features = extractor.execute(image, mask)
        return selected_features(raw_features, label_name)
    except (ValueError, KeyError, RuntimeError) as error:
        print(f"Could not extract features for {label_name}: {error}")
        return {}


def combine_kidneys(right_features, left_features):
    """Combines right and left kidney features by averaging available valid feature values."""
    kidneys = {}
    all_keys = set(right_features.keys()) | set(left_features.keys())
    for key in all_keys:
        values = []
        for features in [right_features, left_features]:
            if key in features:
                value = features[key]
                if value != MISSING_VALUE:
                    values.append(float(value))
        if values:
            kidneys[key.replace("_kidney_right", "_kidneys").replace("_kidney_left", "_kidneys")] = float(np.mean(values))

    return kidneys


def get_feature_value(features, feature_name):
    """Retrieves a feature value and replaces missing or invalid values with -9999."""
    value = features.get(feature_name, MISSING_VALUE)
    if value == MISSING_VALUE:
        return MISSING_VALUE
    return clean_value(value)


def calculate_diff(features, feature_type, organ_1, organ_2):
    """Calculates the difference between the same feature type from two organs."""
    value_1 = get_feature_value(features, f"{feature_type}_{organ_1}")
    value_2 = get_feature_value(features, f"{feature_type}_{organ_2}")
    if value_1 == MISSING_VALUE or value_2 == MISSING_VALUE:
        return MISSING_VALUE
    return value_1 - value_2


def add_difference_features(features):
    """Adds organ-to-organ difference features"""
    diff_specs = [
        ("Mean_diff_aorta_portal_vein_and_splenic_vein", "Mean", "aorta", "portal_vein_and_splenic_vein"),
        ("Median_diff_aorta_liver", "Median", "aorta", "liver"),
        ("Median_diff_kidneys_portal_vein_and_splenic_vein", "Median", "kidneys", "portal_vein_and_splenic_vein"),
        ("Median_diff_aorta_portal_vein_and_splenic_vein", "Median", "aorta", "portal_vein_and_splenic_vein"),
        ("Mean_diff_liver_portal_vein_and_splenic_vein", "Mean", "liver", "portal_vein_and_splenic_vein"),
        ("Median_diff_liver_portal_vein_and_splenic_vein", "Median", "liver", "portal_vein_and_splenic_vein"),
        ("Median_diff_aorta_kidneys", "Median", "aorta", "kidneys"),
        ("Mean_diff_aorta_liver", "Mean", "aorta", "liver"),
        ("10th Percentile_diff_aorta_kidneys", "10th Percentile", "aorta", "kidneys"),
        ("50th Percentile_diff_aorta_kidneys", "50th Percentile", "aorta", "kidneys"),
        ("Mean_diff_liver_spleen", "Mean", "liver", "spleen"),
        ("Mean_diff_kidneys_portal_vein_and_splenic_vein", "Mean", "kidneys", "portal_vein_and_splenic_vein"),
    ]
    for output_name, feature_type, organ_1, organ_2 in diff_specs:
        features[output_name] = calculate_diff(features, feature_type, organ_1, organ_2)
    return features

def extract_organ_features(image, seg_img):
    """Extracts radiomic features for the selected organs from the TotalSegmentator segmentation output."""
    seg_data = np.asanyarray(seg_img.dataobj)
    organ_to_label = {organ_name: label_id for label_id, organ_name in class_map["total"].items()}
    organ_features = {}
    print("\nMask intensity check:")

    for organ in ORGANS:
        if organ not in organ_to_label:
            print(f"Warning: {organ} is not in TotalSegmentator class_map")
            continue
        label_id = organ_to_label[organ]
        mask_np = seg_data == label_id
        if mask_np.sum() == 0:
            print(f"Warning: empty mask for {organ}")
            continue
        mask = nibabel_mask_to_sitk(mask_np, image)
        organ_features[organ] = extract_features_for_mask(image, mask, organ)
    if "kidney_right" in organ_features and "kidney_left" in organ_features:
        organ_features["kidneys"] = combine_kidneys(organ_features["kidney_right"],organ_features["kidney_left"])
    return organ_features

def get_features(input_data, fast=False, model_dir=None):
    """Runs the full feature extraction pipeline, including image reading, clipping, segmentation, radiomics, and final feature selection."""
    original_image = read_ct_image(input_data)
    image_for_features = clip_image(original_image)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "clipped_input.nii.gz")
        sitk.WriteImage(image_for_features, input_path)
        print("Step 1/2: Segmenting organs...")
        seg_img = segment_selected_organs(input_data=input_path,organs=ORGANS,fast=fast,model_dir=model_dir)
        print("Step 2/2: Extracting radiomic features...")
        features_by_organ = extract_organ_features(image_for_features, seg_img)
    features = {}
    for organ_values in features_by_organ.values():
        features.update(organ_values)
    features = add_difference_features(features)
    selected = {feature: features.get(feature, MISSING_VALUE) for feature in FINAL_FEATURES}

    return pd.DataFrame([selected]).replace([np.inf, -np.inf], MISSING_VALUE).fillna(MISSING_VALUE)
