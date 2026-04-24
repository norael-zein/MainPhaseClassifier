"""
Script for running the TotalSegmentator segmentation on the input data for the pipeline.
"""
from totalsegmentator.python_api import totalsegmentator


def segment_selected_organs(input_data, organs, task="total", fast=False):
    if organs is None or len(organs) == 0:
        raise ValueError("You need to input at least one organ.")
    seg_img = totalsegmentator(input=input_data,output=None,task=task,roi_subset=organs,ml=True,skip_saving=True,fast=fast, nr_thr_saving=1,)
    return seg_img
