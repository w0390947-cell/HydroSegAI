#!/usr/bin/env python3
"""
环境检查脚本 - 验证 SAM3 零样本评估所需的环境配置

运行此脚本来检查：
1. Python 依赖是否安装
2. SAM3 模块是否可用
3. Potsdam 数据集是否在正确位置
4. GPU 是否可用
5. 磁盘空间是否充足
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
import platform

def check_python_version():
    """检查 Python 版本"""
    print("🔍 检查 Python 版本...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python 版本: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python 版本过低: {version.major}.{version.minor}.{version.micro}")
        print("   需要 Python 3.8 或更高版本")
        return False

def check_dependencies():
    """检查必要的依赖包"""
    print("\n🔍 检查 Python 依赖...")

    required_packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
        'cv2': 'OpenCV',
        'matplotlib': 'Matplotlib',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn'
    }

    optional_packages = {
        'gdal': 'GDAL (用于读取 GeoTIFF)',
        'sam3': 'SAM3 模块'
    }

    all_required_installed = True

    print("必要依赖:")
    for module, name in required_packages.items():
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - 未安装")
            all_required_installed = False

    print("\n可选依赖:")
    for module, name in optional_packages.items():
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ⚠️  {name} - 未安装（可选）")

    return all_required_installed

def check_gpu():
    """检查 GPU 可用性"""
    print("\n🔍 检查 GPU 状态...")
    if torch.cuda.is_available():
        print(f"✅ CUDA 可用")
        print(f"   GPU 数量: {torch.cuda.device_count()}")
        print(f"   当前 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        return True
    else:
        print("⚠️  CUDA 不可用，将使用 CPU（速度较慢）")
        return False

def check_sam3():
    """检查 SAM3 模块和权重"""
    print("\n🔍 检查 SAM3...")

    try:
        from sam3.model_builder import build_sam3_image_model
        print("✅ SAM3 模块可用")
    except ImportError as e:
        print(f"❌ SAM3 模块导入失败: {e}")
        return False

    # 检查权重文件
    checkpoint_path = Path("/home/anjou/PythonENV/Test_11/sam3/checkpoints/sam3.1_multiplex.pt")
    if checkpoint_path.exists():
        size_gb = checkpoint_path.stat().st_size / (1024**3)
        print(f"✅ SAM3 权重文件存在 (大小: {size_gb:.2f} GB)")
        return True
    else:
        print(f"❌ SAM3 权重文件不存在: {checkpoint_path}")
        return False

def check_potsdam_dataset():
    """检查 Potsdam 数据集"""
    print("\n🔍 检查 Potsdam 数据集...")

    base_dir = Path("/home/anjou/PythonENV/Test_11/Potsdam")

    if not base_dir.exists():
        print(f"❌ Potsdam 目录不存在: {base_dir}")
        return False

    print(f"✅ Potsdam 基础目录存在")

    # 检查必要的子目录
    required_dirs = [
        "2_Ortho_RGB/2_Ortho_RGB",
        "5_Labels_for_participants_no_Boundary/5_Labels_for_participants_no_Boundary"
    ]

    all_dirs_exist = True
    for dir_path in required_dirs:
        full_path = base_dir / dir_path
        if full_path.exists():
            # 统计文件数量
            files = list(full_path.glob("*.tif"))
            print(f"   ✅ {dir_path}: {len(files)} 个文件")
        else:
            print(f"   ❌ {dir_path}: 目录不存在")
            all_dirs_exist = False

    # 检查测试图像
    print("\n   检查测试图像...")
    test_images = [
        "top_potsdam_2_10_RGB.tif",
        "top_potsdam_5_11_RGB.tif",
        "top_potsdam_7_9_RGB.tif"
    ]

    all_images_exist = True
    for image_name in test_images:
        image_path = base_dir / "2_Ortho_RGB/2_Ortho_RGB" / image_name
        if image_path.exists():
            print(f"   ✅ {image_name}")
        else:
            print(f"   ❌ {image_name} - 文件不存在")
            all_images_exist = False

    return all_dirs_exist and all_images_exist

def check_disk_space():
    """检查磁盘空间"""
    print("\n🔍 检查磁盘空间...")

    output_dir = Path("/home/anjou/PythonENV/Test_11/results")

    try:
        # 获取磁盘使用情况
        stat = os.statvfs(output_dir)
        free_space_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        total_space_gb = (stat.f_blocks * stat.f_frsize) / (1024**3)

        print(f"✅ 输出目录磁盘空间:")
        print(f"   可用空间: {free_space_gb:.2f} GB")
        print(f"   总空间: {total_space_gb:.2f} GB")

        if free_space_gb < 5:
            print(f"   ⚠️  警告: 磁盘空间较少（建议至少 5 GB）")
            return False
        else:
            return True

    except Exception as e:
        print(f"⚠️  无法检查磁盘空间: {e}")
        return True

def check_system_info():
    """显示系统信息"""
    print("\n🔍 系统信息:")
    print(f"   操作系统: {platform.system()} {platform.release()}")
    print(f"   Python 版本: {sys.version}")
    print(f"   处理器: {platform.processor()}")

def main():
    """主函数"""
    print("=" * 60)
    print("SAM3 零样本评估 - 环境检查")
    print("=" * 60)

    results = {}

    # 运行所有检查
    results['python_version'] = check_python_version()
    results['dependencies'] = check_dependencies()
    results['gpu'] = check_gpu()
    results['sam3'] = check_sam3()
    results['potsdam_dataset'] = check_potsdam_dataset()
    results['disk_space'] = check_disk_space()
    check_system_info()

    # 总结结果
    print("\n" + "=" * 60)
    print("检查结果总结")
    print("=" * 60)

    critical_checks = ['python_version', 'dependencies', 'sam3', 'potsdam_dataset']
    all_critical_passed = all(results.get(check, False) for check in critical_checks)

    for check_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{check_name.replace('_', ' ').title()}: {status}")

    print("\n" + "=" * 60)

    if all_critical_passed:
        print("🎉 环境检查通过！您可以开始运行 SAM3 零样本评估了。")
        print("\n运行命令:")
        print("  python sam3_potsdam_evaluation.py")
        return 0
    else:
        print("⚠️  环境检查发现问题，请先解决上述问题再运行评估。")

        # 提供安装建议
        if not results.get('dependencies'):
            print("\n💡 安装缺失的依赖:")
            print("  pip install torch torchvision opencv-python matplotlib scikit-learn pandas")

        if not results.get('sam3'):
            print("\n💡 安装 SAM3:")
            print("  cd sam3")
            print("  pip install -e .")

        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)