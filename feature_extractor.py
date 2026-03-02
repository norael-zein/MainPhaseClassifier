"""
Extracting features after segmentation with TotalSegmentator.
"""
import os
import SimpleITK as sitk
import six
from radiomics import featureextractor, getTestCase

def extract_features(image_path, mask_path):
    #Initialize the feature extractor
    extractor = featureextractor.RadiomicsFeatureExtractor()

    #Load the image and mask
    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)

    #Extract features
    features = extractor.execute(image, mask)

    return features

def calculate_feature_differences(organ_features):
    """
    Calculate differences between statistical features from different organs.
    
    Args:
        organ_features: Dict with organ names as keys and feature dicts as values
                       Example: {"liver": {feature_dict}, "aorta": {feature_dict}}
    Returns:
        Dict with differences between organs for each statistical feature
    """
    differences = {}
    organ_names = list(organ_features.keys())
    
    # Get all feature names from the first organ (assuming same features for all organs)
    if not organ_names or not organ_features[organ_names[0]]:
        return differences
    
    feature_names = list(organ_features[organ_names[0]].keys())
    
    # For each feature, calculate differences between organs
    for feature_name in feature_names:
        feature_diff_key = f"{feature_name}_differences"
        differences[feature_diff_key] = {}
        
        # Calculate pairwise differences between organs
        for i in range(len(organ_names)):
            for j in range(i + 1, len(organ_names)):
                organ_i = organ_names[i]
                organ_j = organ_names[j]
                
                feature_i = organ_features[organ_i].get(feature_name)
                feature_j = organ_features[organ_j].get(feature_name)
                
                # Only calculate if both features exist and are numeric
                if feature_i is not None and feature_j is not None:
                    try:
                        diff_value = float(feature_i) - float(feature_j)
                        diff_key = f"{organ_i}_vs_{organ_j}"
                        differences[feature_diff_key][diff_key] = diff_value
                    except (ValueError, TypeError):
                        # Skip non-numeric features
                        pass
    
    return differences


"""
if __name__ == "__main__":
    #Example
    image_path = "Z:/CTphaseClassification/PLC_CECT/ct_files_converted/P0001_ct_C1_converted.nii.gz"

    mask_path1 = "Z:/CTphaseClassification/PLC_CECT/TotalSegmentator/P0001_ct_C1_converted/aorta.nii.gz"
    mask_path2 = "Z:/CTphaseClassification/PLC_CECT/TotalSegmentator/P0001_ct_C1_converted/spleen.nii.gz"

    features1 = extract_features(image_path, mask_path1)
    features2 = extract_features(image_path, mask_path2)
    
    print("Extracted Features for Aorta:")
    for key, value in features1.items():
        print(f"{key}: {value}")

    print("\nExtracted Features for Liver:")
    for key, value in features2.items():
        print(f"{key}: {value}")
    
    # Calculate differences between organs
    organ_features = {
        "aorta": {"original_firstorder_Mean": float(features1["original_firstorder_Mean"])},
        "spleen": {"original_firstorder_Mean": float(features2["original_firstorder_Mean"])},
    }
    differences = calculate_feature_differences(organ_features)
    
    print("\nFeature Differences between Organs:")
    for feature_name, organ_diffs in differences.items():
        print(f"\n{feature_name}:")
        for organ_pair, diff_value in organ_diffs.items():
            print(f"  {organ_pair}: {diff_value}")
"""