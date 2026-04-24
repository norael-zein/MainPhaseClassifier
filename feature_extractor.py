"""
Script used for the feature extraction in the pipeline. 
"""
import SimpleITK as sitk
from radiomics import featureextractor

def extract_features(image_input, mask_input, selected_features=None):
    extractor = featureextractor.RadiomicsFeatureExtractor()

    if isinstance(image_input, str):
        image = sitk.ReadImage(image_input)
    else:
        image = image_input
    if isinstance(mask_input, str):
        mask = sitk.ReadImage(mask_input)
    else:
        mask = mask_input
    features = extractor.execute(image, mask)
    features = {k: v for k, v in features.items() if not k.startswith("diagnostics")}

    if selected_features is not None:
        features = {k: features[k] for k in selected_features if k in features}
    return features

def calculate_feature_differences(organ_features):
    differences = {}
    organ_names = list(organ_features.keys())

    if not organ_names:
        return differences

    feature_names = set()
    for features in organ_features.values():
        feature_names.update(features.keys())

    for feature_name in feature_names:
        feature_diff_key = f"{feature_name}_differences"
        differences[feature_diff_key] = {}

        for i in range(len(organ_names)):
            for j in range(i + 1, len(organ_names)):
                organ_i = organ_names[i]
                organ_j = organ_names[j]

                feature_i = organ_features[organ_i].get(feature_name)
                feature_j = organ_features[organ_j].get(feature_name)

                if feature_i is not None and feature_j is not None:
                    try:
                        diff_value = float(feature_i) - float(feature_j)
                        diff_key = f"{organ_i}_vs_{organ_j}"
                        differences[feature_diff_key][diff_key] = diff_value
                    except (ValueError, TypeError):
                        pass
    return differences