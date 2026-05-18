#!/usr/bin/env python3
"""Exp3-C: SAM3 + Potsdam RGBIR-derived 3-channel composite."""

from exp3_sam3_potsdam_multimodal_common import run_single_modality_experiment


EXPERIMENT_NAME = "Exp3-C: SAM3 Potsdam RGBIR composite"
DEFAULT_CONFIG_PATH = "exp3c_sam3_potsdam_rgbir_composite.yaml"
DEFAULT_OUTPUT_DIR = "/home/anjou/PythonENV/Test_11/results_exp3c_sam3_potsdam_rgbir_composite"


if __name__ == "__main__":
    run_single_modality_experiment(
        EXPERIMENT_NAME,
        DEFAULT_CONFIG_PATH,
        DEFAULT_OUTPUT_DIR,
    )
