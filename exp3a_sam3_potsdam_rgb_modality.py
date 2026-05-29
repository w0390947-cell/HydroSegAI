#!/usr/bin/env python3

from exp3_sam3_potsdam_multimodal_common import run_single_modality_experiment


EXPERIMENT_NAME = "Exp3-A: SAM3 Potsdam RGB modality"
DEFAULT_CONFIG_PATH = "exp3a_sam3_potsdam_rgb_modality.yaml"
DEFAULT_OUTPUT_DIR = "/home/anjou/PythonENV/Test_11/results_exp3a_sam3_potsdam_rgb_modality_dev"


if __name__ == "__main__":
    run_single_modality_experiment(
        EXPERIMENT_NAME,
        DEFAULT_CONFIG_PATH,
        DEFAULT_OUTPUT_DIR,
    )
