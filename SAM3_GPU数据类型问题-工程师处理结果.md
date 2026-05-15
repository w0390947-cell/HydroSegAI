# SAM3 GPU 数据类型问题 - 工程师处理结果

## 处理结论

已按“不使用 CPU 运算”的要求处理。当前方案是在 GPU 上显式统一 SAM3 模型权重 dtype，并在 `set_image()` 与 `set_text_prompt()` 推理阶段使用相同 dtype 的 `torch.autocast`，避免出现：

```text
mat1 and mat2 must have the same dtype, but got BFloat16 and Float
```

## 修改文件

1. `sam3_potsdam_evaluation.py`
2. `config_sam3_evaluation.yaml`
3. `test_gpu_dtype.py`

## 核心修改

### 1. 新增 GPU dtype 配置

`config_sam3_evaluation.yaml` 的 `device` 段新增：

```yaml
gpu_dtype: "auto"
```

支持值：

```text
auto
bfloat16
float16
float32
```

`auto` 策略：

- CUDA 可用且支持 BF16 时，使用 `torch.bfloat16`。
- 不支持 BF16 时，使用 `torch.float16`。
- 不会回退到 CPU。

### 2. 模型加载后统一 dtype

`PotsdamSAM3Evaluator._configure_cuda_precision()` 会在模型加载后执行：

```python
self.model.to(device=self.device, dtype=self.model_dtype)
```

并记录首个参数的 dtype 和 device，便于服务器侧确认模型确实在 GPU 上运行。

### 3. 推理阶段启用同 dtype autocast

`predict_patch()` 中对以下两个阶段加了统一上下文：

```python
with torch.inference_mode(), self._inference_autocast():
    inference_state = self.processor.set_image(pil_image)

with torch.inference_mode(), self._inference_autocast():
    output = self.processor.set_text_prompt(...)
```

这样可以避免 SAM3 内部 BF16 激活与 Float32 权重混用。

## 服务器侧验证命令

建议先运行 smoke test：

```bash
cd /home/anjou/PythonENV/Test_11
source .venv_hf/bin/activate
python test_gpu_dtype.py
```

预期重点输出：

```text
CUDA available: True
BF16 supported: True
Configured model dtype: torch.bfloat16
First parameter dtype: torch.bfloat16
First parameter device: cuda:0
GPU dtype smoke test finished without dtype exception.
```

如果仍出现 dtype 不兼容，请临时强制 BF16：

```yaml
device:
  gpu_dtype: "bfloat16"
```

然后重新运行：

```bash
python test_gpu_dtype.py
```

## 正式运行

smoke test 通过后运行：

```bash
python sam3_potsdam_evaluation.py --config config_sam3_evaluation.yaml
```

## 注意

本次修复不采用 CPU fallback。若服务器日志显示 `设备: CPU`，应优先检查 CUDA/PyTorch 环境，而不是通过脚本降级运行。
