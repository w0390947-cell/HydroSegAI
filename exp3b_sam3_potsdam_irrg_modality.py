#!/usr/bin/env python3
"""Exp3-B: SAM3 + Potsdam IRRG under fixed Exp2-C reasoning strategy."""

from exp3_sam3_potsdam_multimodal_common import run_single_modality_experiment


EXPERIMENT_NAME = "Exp3-B: SAM3 Potsdam IRRG modality"
DEFAULT_CONFIG_PATH = "exp3b_sam3_potsdam_irrg_modality.yaml"
DEFAULT_OUTPUT_DIR = "/home/anjou/PythonENV/Test_11/results_exp3b_sam3_potsdam_irrg_modality"


if __name__ == "__main__":
    run_single_modality_experiment(
        EXPERIMENT_NAME,
        DEFAULT_CONFIG_PATH,
        DEFAULT_OUTPUT_DIR,
    )
