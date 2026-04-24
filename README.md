# CT Phase Classification Pipeline

This project provides a full pipeline for automatic CT phase classification using organ segmentation, radiomics feature extraction, and a trained machine learning model.

The pipeline uses the TotalSegmentator framework (Wasserthal et al. 2023) to segment the following organs:

- Aorta  
- Portal vein and splenic vein  
- Urinary bladder  
- Kidneys (left and right)  
- Spleen  
- Liver  

Radiomics features extracted from these organs are used as input to an XGBoost classifier, which predicts the CT phase.

---

## Predicted CT Phases

The model classifies scans into the following phases:

- **NP** : Non-Contrast Phase  
- **AP** : Arterial Phase  
- **VP** : Portal-Venous Phase  
- **DP** : Delayed Phase  

---

## Requirements

This project requires **Python 3.10 or 3.11** (PyRadiomics is not compatible with Python 3.12+).

---

## How to Run

### 1. Install packages
```bash
pip install -r requirements.txt
```

### 2. Run the pipeline
```bash
python pipeline.py --input "input_path" --output "output_folder"
```
The input_path could either be a DICOM file or NiFTI file. 


## References

- TotalSegmentator: https://github.com/wasserth/TotalSegmentator  
- Wasserthal et al., *TotalSegmentator: Robust segmentation of 104 anatomical structures in CT images*, Radiology: AI (2023)