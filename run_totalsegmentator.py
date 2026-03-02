import nibabel as nib
from totalsegmentator.python_api import totalsegmentator


def segment_selected_organs(input_data, output_path, organs, task="total", fast=False):
    """
    Segment out selected organs with TotalSegmentator.

    Parameters:
    input_data : NIfTI-file or DICOM files
    output_path : folder where the segmented output will be saved in NIfTI-format
    organs : list[str], list with organs
    task : Task-name, Default is "total". 
    fast: If True, the model will be run in fast mode, which is faster but less accurate. Default is False.
    See https://github.com/wasserth/TotalSegmentator
    """

    if organs is None or len(organs) == 0:
        raise ValueError("You need to input at least one organ.")

    result = totalsegmentator(input_data, output_path, task=task, roi_subset=organs)

    return result

if __name__ == "__main__":
    input_data = "P0059_ct_C1_voxel_converted.nii.gz"
    output_path = "Output_folder"
    organs = ["liver", "kidney_right"]  

    segment_selected_organs(input_data, output_path, organs)
