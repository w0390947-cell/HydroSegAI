#!/usr/bin/env python3
"""Smoke test for SAM3 GPU dtype compatibility."""

import sys

import numpy as np
import torch

from sam3_potsdam_evaluation import PotsdamSAM3Evaluator


def main():
    print("=== SAM3 GPU dtype smoke test ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This test must run on GPU.")
        return 1

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")

    evaluator = PotsdamSAM3Evaluator(
        base_dir="/home/anjou/PythonENV/Test_11/Potsdam",
        output_dir="/tmp/sam3_gpu_dtype_test",
        checkpoint_path="/home/anjou/PythonENV/Test_11/sam3/checkpoints/sam3.1_multiplex.pt",
        gpu_dtype="auto",
    )

    first_param = next(evaluator.model.parameters())
    print(f"Evaluator device: {evaluator.device}")
    print(f"Configured model dtype: {evaluator.model_dtype}")
    print(f"First parameter dtype: {first_param.dtype}")
    print(f"First parameter device: {first_param.device}")

    test_patch = np.random.randint(0, 255, (1008, 1008, 3), dtype=np.uint8)
    predictions = evaluator.predict_patch(test_patch, ["building"])
    print(f"Returned masks: {len(predictions.get('masks', []))}")
    print("GPU dtype smoke test finished without dtype exception.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
