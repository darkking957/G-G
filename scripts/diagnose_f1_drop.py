# scripts/diagnose_f1_drop.py
"""
诊断F1分数下降的原因
"""

import json
import argparse
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import numpy as np


def load_predictions(file_path: str) -> List[Dict]:
    """加载预测结果"""
    predictions = []
    with open(file_path, 'r') as f:
        for line in f:
            predictions.append(json.loads(line))
    return predictions


def analyze_path_differences(baseline_preds: List[Dict], got_preds: List[Dict]) -> Dict:
    """分析路径差异"""
    analysis = {
        "total_samples": len(baseline_preds),
        "path_changes": 0,
        "path_losses": 0,
        "path_additions": 0,
        "quality_degradation": [],
        "missing_ground_truth": []
    }
    
    # 创建ID映射
    baseline_map = {p["id"]: p for p in baseline_preds}
    got_map = {p["id"]: p for p in got_preds}
    
    for qid in baseline_map:
        if qid not in got_map:
            continue
            
        baseline = baseline_map[qid]
        got = got_map[qid]
        
        # 比较路径数量
        baseline_paths = set(tuple(p) for p in baseline["prediction"])
        got_paths = set(tuple(p) for p in got["prediction"])
        ground_paths = set(tuple(p) for p in baseline["ground_paths"])
        
        if baseline_paths != got_paths:
            analysis["path_changes"] += 1
            
        # 检查是否丢失了有效路径
        lost_paths = baseline_paths - got_paths
        if lost_paths:
            analysis["path_losses"] += 1
            
            # 检查丢失的路径是否包含ground truth
            lost_ground_truth = lost_paths & ground_paths
            if lost_ground_truth:
                analysis["missing_ground_truth"].append({
                    "id": qid,
                    "question": baseline["question"],
                    "lost_paths": [list(p) for p in lost_ground_truth]
                })
                
        # 检查新增路径
        added_paths = got_paths - baseline_paths
        if added_paths:
            analysis["path_additions"] += 1
            
        # 检查质量下降
        if len(got_paths) > len(baseline_paths) * 2:
            analysis["quality_degradation"].append({
                "id": qid,
                "baseline_count": len(baseline_paths),
                "got_count": len(got_paths),
                "ratio": len(got_paths) / len(baseline_paths)
            })
            
    return analysis


def analyze_got_specific_issues(got_preds: List[Dict]) -> Dict:
    """分析GoT特定问题"""
    issues = {
        "validation_failures": 0,
        "enhancement_failures": 0,
        "low_confidence_enhancements": [],
        "excessive_iterations": [],
        "time_issues": []
    }
    
    for pred in got_preds:
        if "got_info" not in pred:
            continue
            
        got_info = pred["got_info"]
        
        # 检查验证问题
        if "stats" in got_info:
            stats = got_info["stats"]
            if stats["validated_count"] < stats["initial_count"] * 0.5:
                issues["validation_failures"] += 1
                
        # 检查增强效果
        if got_info.get("enhanced_count", 0) == got_info.get("original_count", 0):
            issues["enhancement_failures"] += 1
            
        # 检查时间问题
        if "time" in got_info and got_info["time"] > 1.0:
            issues["time_issues"].append({
                "id": pred["id"],
                "time": got_info["time"]
            })
            
        # 检查诊断信息
        if "diagnosis" in got_info and got_info["diagnosis"]["issues"]:
            for issue in got_info["diagnosis"]["issues"]:
                if "Missing" in issue:
                    issues["validation_failures"] += 1
                    
    return issues


def generate_diagnosis_report(baseline_file: str, got_file: str, output_file: str):
    """生成诊断报告"""
    print("Loading predictions...")
    baseline_preds = load_predictions(baseline_file)
    got_preds = load_predictions(got_file)
    
    print("Analyzing differences...")
    path_analysis = analyze_path_differences(baseline_preds, got_preds)
    got_issues = analyze_got_specific_issues(got_preds)
    
    # 生成报告
    with open(output_file, 'w') as f:
        f.write("F1 Score Drop Diagnosis Report\n")
        f.write("=" * 50 + "\n\n")
        
        # 路径分析
        f.write("Path Analysis:\n")
        f.write(f"- Total samples: {path_analysis['total_samples']}\n")
        f.write(f"- Samples with path changes: {path_analysis['path_changes']}\n")
        f.write(f"- Samples with path losses: {path_analysis['path_losses']}\n")
        f.write(f"- Samples with path additions: {path_analysis['path_additions']}\n")
        f.write(f"- Samples missing ground truth: {len(path_analysis['missing_ground_truth'])}\n")
        f.write("\n")
        
        # GoT问题
        f.write("GoT-Specific Issues:\n")
        f.write(f"- Validation failures: {got_issues['validation_failures']}\n")
        f.write(f"- Enhancement failures: {got_issues['enhancement_failures']}\n")
        f.write(f"- Time issues: {len(got_issues['time_issues'])}\n")
        f.write("\n")
        
        # 根本原因分析
        f.write("Root Cause Analysis:\n")
        if path_analysis['path_losses'] > path_analysis['total_samples'] * 0.1:
            f.write("⚠️  HIGH PATH LOSS RATE - Validation is too strict\n")
            
        if len(path_analysis['missing_ground_truth']) > 0:
            f.write("⚠️  GROUND TRUTH PATHS LOST - Critical validation error\n")
            
        if len(path_analysis['quality_degradation']) > path_analysis['total_samples'] * 0.2:
            f.write("⚠️  QUALITY DEGRADATION - Too many noisy paths added\n")
            
        if got_issues['validation_failures'] > path_analysis['total_samples'] * 0.1:
            f.write("⚠️  VALIDATION FAILURES - Need to adjust validation logic\n")
            
        f.write("\n")
        
        # 建议
        f.write("Recommendations:\n")
        f.write("1. Set --got_preserve_original=True to keep all original paths\n")
        f.write("2. Reduce --got_confidence_threshold to 0.9 for stricter enhancement\n")
        f.write("3. Set --got_max_enhanced=2 to limit noise\n")
        f.write("4. Consider disabling validation with --got_enable_validation=False\n")
        f.write("\n")
        
        # 具体案例
        if path_analysis['missing_ground_truth']:
            f.write("Examples of Lost Ground Truth Paths:\n")
            for example in path_analysis['missing_ground_truth'][:3]:
                f.write(f"\nQuestion: {example['question']}\n")
                f.write(f"Lost paths: {example['lost_paths']}\n")


def create_fixed_config(diagnosis_file: str, output_file: str):
    """基于诊断结果创建修复配置"""
    # 读取诊断结果
    with open(diagnosis_file, 'r') as f:
        content = f.read()
        
    # 生成优化配置
    config = {
        "use_got": True,
        "got_preserve_original": True,  # 关键：保留原始路径
        "got_enable_validation": False if "VALIDATION FAILURES" in content else True,
        "got_enable_enhancement": True,
        "got_confidence_threshold": 0.9 if "QUALITY DEGRADATION" in content else 0.8,
        "got_max_enhanced": 2 if "Too many noisy paths" in content else 3,
    }
    
    # 保存配置
    with open(output_file, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Auto-generated fixed configuration\n\n")
        
        for key, value in config.items():
            if isinstance(value, bool):
                if value:
                    f.write(f"--{key} \\\n")
            else:
                f.write(f"--{key} {value} \\\n")
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, required=True,
                       help="Baseline predictions file")
    parser.add_argument("--got", type=str, required=True,
                       help="GoT predictions file")
    parser.add_argument("--output", type=str, default="f1_diagnosis.txt",
                       help="Output diagnosis file")
    parser.add_argument("--create_fix", action="store_true",
                       help="Create fixed configuration")
    
    args = parser.parse_args()
    
    # 生成诊断报告
    generate_diagnosis_report(args.baseline, args.got, args.output)
    print(f"Diagnosis report saved to: {args.output}")
    
    # 生成修复配置
    if args.create_fix:
        fix_config = args.output.replace(".txt", "_fix.sh")
        create_fixed_config(args.output, fix_config)
        print(f"Fixed configuration saved to: {fix_config}")