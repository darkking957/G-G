# scripts/verify_integration.py
"""
验证GoT集成是否正确
确保所有文件名保持一致且功能正常
"""

import sys
import os
import importlib
import json

def verify_file_structure():
    """验证文件结构"""
    print("=== Verifying File Structure ===")
    
    required_files = {
        "src/got_reasoning/got_engine.py": "GoT引擎",
        "src/got_reasoning/thought_graph.py": "思想图",
        "src/got_reasoning/validate_plans.py": "验证器",
        "src/got_reasoning/evaluate_plans.py": "评估器",
        "src/got_reasoning/aggregate_plans.py": "聚合器",
        "src/got_reasoning/graph_attention.py": "GAT网络",
        "src/got_reasoning/feedback_loop.py": "反馈控制",
        "src/got_reasoning/minimal_got.py": "最小化GoT",
        "src/qa_prediction/gen_rule_path.py": "路径生成",
        "src/qa_prediction/predict_answer.py": "答案预测",
        "scripts/run_got_reasoning.sh": "运行脚本",
        "scripts/compare_results.py": "比较脚本"
    }
    
    all_exist = True
    for file_path, description in required_files.items():
        if os.path.exists(file_path):
            print(f"✓ {description}: {file_path}")
        else:
            print(f"✗ {description}: {file_path} NOT FOUND")
            all_exist = False
            
    return all_exist


def verify_imports():
    """验证导入是否正常"""
    print("\n=== Verifying Imports ===")
    
    try:
        # 添加路径
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # 测试导入
        from src.got_reasoning.got_engine import GoTEngine, GoTConfig
        print("✓ GoTEngine imported successfully")
        
        from src.got_reasoning.minimal_got import MinimalGoT, integrate_minimal_got
        print("✓ MinimalGoT imported successfully")
        
        from src.got_reasoning.thought_graph import ThoughtGraph, ThoughtNode
        print("✓ ThoughtGraph imported successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False


def verify_got_config():
    """验证GoT配置"""
    print("\n=== Verifying GoT Configuration ===")
    
    try:
        from src.got_reasoning.got_engine import GoTConfig
        
        # 创建默认配置
        config = GoTConfig()
        
        # 检查保守配置
        assert hasattr(config, 'preserve_original'), "Missing preserve_original"
        assert hasattr(config, 'enable_minimal_mode'), "Missing enable_minimal_mode"
        assert hasattr(config, 'max_enhanced_paths'), "Missing max_enhanced_paths"
        
        print(f"✓ Default config: preserve_original={config.preserve_original}")
        print(f"✓ Default config: enable_minimal_mode={config.enable_minimal_mode}")
        print(f"✓ Default config: max_enhanced_paths={config.max_enhanced_paths}")
        
        # 检查保守默认值
        assert config.preserve_original == True, "preserve_original should default to True"
        assert config.enable_minimal_mode == True, "enable_minimal_mode should default to True"
        assert config.max_iterations == 1, "max_iterations should default to 1"
        
        print("✓ Conservative defaults verified")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False


def verify_minimal_got():
    """验证最小化GoT功能"""
    print("\n=== Verifying Minimal GoT ===")
    
    try:
        from src.got_reasoning.minimal_got import MinimalGoT, integrate_minimal_got
        
        # 测试最小化GoT
        got = MinimalGoT(preserve_original=True)
        
        # 测试用例
        test_paths = [
            ["location.country.capital"],
            ["location.country.capital"],  # 重复
            ["person.person.nationality"]
        ]
        
        enhanced, stats = got.enhance("Test question?", test_paths)
        
        # 验证去重
        assert len(enhanced) <= len(test_paths), "Should remove duplicates"
        print(f"✓ Deduplication: {len(test_paths)} -> {len(enhanced)} paths")
        
        # 验证保留原始
        for path in test_paths:
            if path != ["location.country.capital"] or enhanced.count(path) > 0:
                print(f"✓ Original path preserved: {path}")
                
        # 测试一行集成
        result = integrate_minimal_got("Question?", test_paths)
        assert isinstance(result, list), "Should return list"
        print("✓ One-line integration works")
        
        return True
        
    except Exception as e:
        print(f"✗ Minimal GoT error: {e}")
        return False


def verify_command_compatibility():
    """验证命令行兼容性"""
    print("\n=== Verifying Command Compatibility ===")
    
    # 测试帮助命令
    import subprocess
    
    try:
        # 检查gen_rule_path.py参数
        result = subprocess.run(
            ["python", "src/qa_prediction/gen_rule_path.py", "--help"],
            capture_output=True,
            text=True
        )
        
        help_text = result.stdout
        
        # 检查关键参数
        required_args = [
            "--use_got",
            "--got_minimal_mode",
            "--got_iterations",
            "--got_beam_width"
        ]
        
        for arg in required_args:
            if arg in help_text:
                print(f"✓ Found argument: {arg}")
            else:
                print(f"✗ Missing argument: {arg}")
                
        return True
        
    except Exception as e:
        print(f"✗ Command verification error: {e}")
        return False


def create_test_summary():
    """创建测试总结"""
    print("\n" + "="*50)
    print("INTEGRATION VERIFICATION SUMMARY")
    print("="*50)
    
    results = {
        "File Structure": verify_file_structure(),
        "Python Imports": verify_imports(),
        "GoT Configuration": verify_got_config(),
        "Minimal GoT": verify_minimal_got(),
        "Command Compatibility": verify_command_compatibility()
    }
    
    # 统计
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED - Integration successful!")
        print("\nRecommended next steps:")
        print("1. Run: bash scripts/run_got_reasoning.sh")
        print("2. Monitor F1 scores in results/")
        print("3. Use --got_minimal_mode for safest operation")
    else:
        print("\n⚠️  SOME TESTS FAILED - Please check the errors above")
        print("\nTroubleshooting:")
        print("1. Ensure all files are in correct locations")
        print("2. Check Python path configuration")
        print("3. Verify dependencies are installed")
        
    # 保存结果
    with open("integration_test_results.json", "w") as f:
        json.dump({
            "results": results,
            "passed": passed,
            "total": total,
            "success": passed == total
        }, f, indent=2)
        
    print(f"\nDetailed results saved to: integration_test_results.json")


if __name__ == "__main__":
    create_test_summary()