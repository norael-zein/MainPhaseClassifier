"""
Script for running TotalSegmentator segmentation on pipeline input data.
"""
import os
from contextlib import contextmanager
from totalsegmentator.python_api import totalsegmentator

@contextmanager
def totalseg_model_dir(model_dir=None):
    """Select the directory to point at"""
    old_value = os.environ.get("TOTALSEG_HOME_DIR")
    if model_dir is None:
        os.environ.pop("TOTALSEG_HOME_DIR", None)
    else:
        os.environ["TOTALSEG_HOME_DIR"] = str(model_dir)
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop("TOTALSEG_HOME_DIR", None)
        else:
            os.environ["TOTALSEG_HOME_DIR"] = old_value

def run_totalsegmentator(input_data,task,roi_subset=None,fast=False,model_dir=None):
    """Function for running the TotalSegmentator task"""
    print(f"Running TotalSegmentator for the {task} task...")
    with totalseg_model_dir(model_dir):
        try:
            return totalsegmentator(input=input_data,output=None,task=task,roi_subset=roi_subset,ml=True,skip_saving=True,fast=fast)
        except Exception as e:
            raise RuntimeError(
                f"TotalSegmentator failed for task '{task}'.\n\n"
                "Possible reasons:\n"
                "- The model weights are not downloaded yet.\n"
                "- TOTALSEG_HOME_DIR points to the wrong folder.\n"
                "- This task may require a TotalSegmentator license.\n\n"
                "Try one of these:\n"
                f"  totalseg_download_weights -t {task}\n"
                "  totalseg_set_license -l YOUR_LICENSE_NUMBER\n"
                "  set TOTALSEG_HOME_DIR to your local model folder\n\n"
                f"Original error:\n{e}"
            ) from e

def segment_selected_organs(input_data,organs=None,task="total",fast=False,model_dir=None):
    """ This function segments out liver, spleen, both kidneys, aorta, portal veind and splenic vein, and the urinary bladder"""
    if organs is None:
        organs = ["liver","spleen","kidney_right","kidney_left","aorta","portal_vein_and_splenic_vein","urinary_bladder"]
    selected_organs = run_totalsegmentator(input_data=input_data,task=task,roi_subset=organs,fast=fast,model_dir=model_dir,)
    return selected_organs

#def segment_tissue(input_data,task="tissue_types",fast=False,model_dir=None):
#      """This function can be used to segment out tissue parts if needed"""  
#    tissues = run_totalsegmentator(input_data=input_data,task=task,fast=fast,model_dir=model_dir)
#    return tissues

#def segment_muscles(input_data,task="abdominal_muscles",fast=False,model_dir=None,):
#    """This can be used to segment muscle parts if needed"""
#    muscles = run_totalsegmentator(input_data=input_data,task=task,fast=fast,model_dir=model_dir)
#    return muscles
