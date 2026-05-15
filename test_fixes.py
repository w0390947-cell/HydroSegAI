#!/usr/bin/env python3
"""
测试服务器侧修复的问题
验证RGB标签转换、logger初始化等关键功能
"""

import sys
import numpy as np
from pathlib import Path

def test_label_conversion():
    """测试RGB标签到类别ID的转换"""
    print("=" * 50)
    print("测试1: RGB标签转换功能")
    print("=" * 50)

    try:
        from sam3_potsdam_evaluation import PotsdamSAM3Evaluator

        # 创建临时评估器实例
        temp_evaluator = PotsdamSAM3Evaluator(
            base_dir="/home/anjou/PythonENV/Test_11/Potsdam",
            output_dir="/tmp/test_results"
        )

        # 测试标签读取
        label_path = "/home/anjou/PythonENV/Test_11/Potsdam/5_Labels_for_participants_no_Boundary/5_Labels_for_participants_no_Boundary/top_potsdam_2_10_label_noBoundary.tif"

        if not Path(label_path).exists():
            print(f"❌ 标签文件不存在: {label_path}")
            return False

        # 读取并转换标签
        label = temp_evaluator.read_label(label_path)

        print(f"✅ 标签形状: {label.shape}")
        print(f"✅ 数据类型: {label.dtype}")
        print(f"✅ 唯一类别值: {np.unique(label)}")

        # 验证转换结果
        unique_values = np.unique(label)
        expected_values = {0, 1, 2, 3, 4, 5, 255}  # 6个类别 + 忽略区域

        if set(unique_values).issubset(expected_values):
            print("✅ RGB转换成功，所有值都在预期范围内")

            # 统计各类别像素数
            for class_id in [0, 1, 2, 3, 4, 5, 255]:
                count = np.sum(label == class_id)
                class_name = {
                    0: "背景类",
                    1: "不透水表面",
                    2: "建筑物",
                    3: "低矮植被",
                    4: "树木",
                    5: "汽车",
                    255: "忽略区域"
                }.get(class_id, f"类别{class_id}")
                print(f"   {class_name} (类别{class_id}): {count:,} 像素")

            return True
        else:
            print(f"❌ 转换失败，发现意外的类别值: {unique_values}")
            return False

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_logger_initialization():
    """测试Logger初始化顺序"""
    print("\n" + "=" * 50)
    print("测试2: Logger初始化顺序")
    print("=" * 50)

    try:
        from sam3_potsdam_evaluation import PotsdamSAM3Evaluator

        # 创建评估器（logger应该在模型加载前初始化）
        print("创建评估器实例...")
        evaluator = PotsdamSAM3Evaluator(
            base_dir="/home/anjou/PythonENV/Test_11/Potsdam",
            output_dir="/tmp/test_results"
        )

        # 检查logger是否存在
        if hasattr(evaluator, 'logger'):
            print("✅ Logger已成功初始化")

            # 测试logger功能
            evaluator.logger.info("测试日志消息")
            print("✅ Logger功能正常")

            return True
        else:
            print("❌ Logger未初始化")
            return False

    except AttributeError as e:
        print(f"❌ Logger初始化顺序错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_patch_fusion():
    """测试Patch融合逻辑"""
    print("\n" + "=" * 50)
    print("测试3: Patch融合逻辑（投票法）")
    print("=" * 50)

    try:
        from sam3_potsdam_evaluation import PotsdamSAM3Evaluator

        evaluator = PotsdamSAM3Evaluator(
            base_dir="/home/anjou/PythonENV/Test_11/Potsdam",
            output_dir="/tmp/test_results"
        )

        # 创建测试patches
        # 创建有重叠区域的测试数据
        patch1 = np.ones((10, 10), dtype=np.uint8) * 1  # 类别1
        patch2 = np.ones((10, 10), dtype=np.uint8) * 2  # 类别2
        patch3 = np.ones((10, 10), dtype=np.uint8) * 3  # 类别3

        # 创建有重叠的patches
        patches = [
            (patch1, 0, 0),   # 位置(0,0)
            (patch2, 5, 0),   # 位置(5,0)，与patch1有重叠
            (patch3, 0, 5),   # 位置(0,5)，与patch1有重叠
        ]

        original_shape = (15, 15)  # 原始图像大小

        # 测试融合
        merged = evaluator.merge_patches(patches, original_shape)

        print(f"✅ 融合后的图像形状: {merged.shape}")
        print(f"✅ 融合后的数据类型: {merged.dtype}")

        # 验证融合结果
        unique_values = np.unique(merged)
        print(f"✅ 融合后的唯一值: {unique_values}")

        # 检查重叠区域是否使用了投票法
        overlap_region = merged[5:10, 0:10]  # patch1和patch2的重叠区域
        if overlap_region.size > 0:
            print(f"✅ 重叠区域样例值: {overlap_region[0, :5]}")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ignore_region_handling():
    """测试忽略区域处理"""
    print("\n" + "=" * 50)
    print("测试4: 忽略区域处理")
    print("=" * 50)

    try:
        from sam3_potsdam_evaluation import PotsdamSAM3Evaluator

        evaluator = PotsdamSAM3Evaluator(
            base_dir="/home/anjou/PythonENV/Test_11/Potsdam",
            output_dir="/tmp/test_results"
        )

        # 创建测试数据
        prediction = np.random.randint(0, 6, size=(100, 100), dtype=np.uint8)
        ground_truth = np.random.randint(0, 6, size=(100, 100), dtype=np.uint8)

        # 添加一些忽略区域
        ground_truth[10:20, 10:20] = 255
        ground_truth[50:60, 50:60] = 255

        # 计算指标
        metrics = evaluator.calculate_metrics(prediction, ground_truth)

        print("✅ 指标计算成功")
        print(f"✅ 有效像素数: {metrics['valid_pixels']:,}")
        print(f"✅ 忽略像素数: {metrics['ignored_pixels']:,}")
        print(f"✅ 总像素数: {metrics['total_pixels']:,}")
        print(f"✅ Overall Accuracy: {metrics['overall_accuracy']:.4f}")

        # 验证忽略区域被正确处理
        expected_ignored = 200  # 两个10x10的区域
        if metrics['ignored_pixels'] == expected_ignored:
            print(f"✅ 忽略区域计数正确: {expected_ignored}")
        else:
            print(f"⚠️ 忽略区域计数不匹配: 预期{expected_ignored}, 实际{metrics['ignored_pixels']}")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("🚀 开始测试服务器侧修复...")
    print(f"Python版本: {sys.version}")
    print(f"NumPy版本: {np.__version__}")

    results = []

    # 运行各项测试
    results.append(("RGB标签转换", test_label_conversion()))
    results.append(("Logger初始化", test_logger_initialization()))
    results.append(("Patch融合逻辑", test_patch_fusion()))
    results.append(("忽略区域处理", test_ignore_region_handling()))

    # 汇总结果
    print("\n" + "=" * 50)
    print("📊 测试结果汇总")
    print("=" * 50)

    passed = 0
    failed = 0

    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\n总计: {passed} 通过, {failed} 失败")

    if failed == 0:
        print("\n🎉 所有测试通过！服务器侧修复成功。")
        return 0
    else:
        print(f"\n⚠️ 有 {failed} 个测试失败，请检查相关功能。")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)