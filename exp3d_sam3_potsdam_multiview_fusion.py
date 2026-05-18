#!/usr/bin/env python3
"""Exp3-D: RGB + IRRG + RGBIR composite multi-view SAM3 score fusion."""

from exp3_sam3_potsdam_multimodal_common import run_multiview_experiment


EXPERIMENT_NAME = "Exp3-D: SAM3 Potsdam multi-view fusion"
DEFAULT_CONFIG_PATH = "exp3d_sam3_potsdam_multiview_fusion.yaml"
DEFAULT_OUTPUT_DIR = "/home/anjou/PythonENV/Test_11/results_exp3d_sam3_potsdam_multiview_fusion"


if __name__ == "__main__":
    run_multiview_experiment(
        EXPERIMENT_NAME,
        DEFAULT_CONFIG_PATH,
        DEFAULT_OUTPUT_DIR,
    )
