# src/qa_prediction/predict_answer.py
# 修改版本：更好地集成GoT结果

import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

import utils
import argparse
from tqdm import tqdm
from llms.language_models import get_registed_model
import os
from datasets import load_dataset
from qa_prediction.evaluate_results import eval_result
import json
from multiprocessing import Pool
from qa_prediction.build_qa_input import PromptBuilder
from functools import partial
import logging

logger = logging.getLogger(__name__)


def get_output_file(path, force=False):
    if not os.path.exists(path) or force:
        fout = open(path, "w")
        return fout, []
    else:
        with open(path, "r") as f:
            processed_results = []
            for line in f:
                try:
                    results = json.loads(line)
                except:
                    raise ValueError("Error in line: ", line)
                processed_results.append(results["id"])
        fout = open(path, "a")
        return fout, processed_results


def merge_rule_result_with_got(qa_dataset, rule_dataset, n_proc=1, filter_empty=False):
    """合并GoT增强的规则结果"""
    question_to_rule = dict()
    
    for data in rule_dataset:
        qid = data["id"]
        predicted_paths = data["prediction"]
        ground_paths = data["ground_paths"]
        
        # 提取GoT信息（如果存在）
        got_info = data.get("got_info", None)
        
        question_to_rule[qid] = {
            "predicted_paths": predicted_paths,
            "ground_paths": ground_paths,
            "got_info": got_info
        }

    def find_rule(sample):
        qid = sample["id"]
        sample["predicted_paths"] = []
        sample["ground_paths"] = []
        sample["got_info"] = None
        
        if qid in question_to_rule:
            sample["predicted_paths"] = question_to_rule[qid]["predicted_paths"]
            sample["ground_paths"] = question_to_rule[qid]["ground_paths"]
            sample["got_info"] = question_to_rule[qid]["got_info"]
            
        return sample

    qa_dataset = qa_dataset.map(find_rule, num_proc=n_proc)
    
    if filter_empty:
        qa_dataset = qa_dataset.filter(
            lambda x: len(x["ground_paths"]) > 0, num_proc=n_proc
        )
        
    return qa_dataset


class GoTEnhancedPromptBuilder(PromptBuilder):
    """GoT增强的提示构建器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def process_input_with_got(self, question_dict):
        """处理包含GoT信息的输入"""
        # 首先使用基类方法构建基本输入
        base_input = self.process_input(question_dict)
        
        # 如果有GoT信息，添加额外的上下文
        got_info = question_dict.get("got_info", None)
        if got_info and self.add_rule:
            # 添加GoT增强信息
            got_context = self._format_got_context(got_info)
            
            # 在推理路径后添加GoT上下文
            if "Reasoning Paths:" in base_input:
                base_input = base_input.replace(
                    "Question:",
                    f"{got_context}\n\nQuestion:"
                )
                
        return base_input
        
    def _format_got_context(self, got_info):
        """格式化GoT上下文信息"""
        context_parts = []
        
        if "best_score" in got_info:
            context_parts.append(f"Reasoning Confidence: {got_info['best_score']:.2f}")
            
        if "iterations" in got_info:
            context_parts.append(f"Reasoning Iterations: {got_info['iterations']}")
            
        if context_parts:
            return "**GoT Analysis:**\n" + "\n".join(context_parts)
        else:
            return ""


def prediction_with_got(data, processed_list, input_builder, model):
    """支持GoT的预测函数"""
    question = data["question"]
    answer = data["answer"]
    id = data["id"]
    
    if id in processed_list:
        return None
        
    # 检查是否有GoT信息
    got_info = data.get("got_info", None)
    
    if model is None:
        prediction = input_builder.direct_answer(data)
        return {
            "id": id,
            "question": question,
            "prediction": prediction,
            "ground_truth": answer,
            "input": question,
            "got_enhanced": got_info is not None
        }
        
    # 使用GoT增强的输入构建
    if isinstance(input_builder, GoTEnhancedPromptBuilder):
        input_text = input_builder.process_input_with_got(data)
    else:
        input_text = input_builder.process_input(data)
        
    prediction = model.generate_sentence(input_text)
    
    if prediction is None:
        return None
        
    result = {
        "id": id,
        "question": question,
        "prediction": prediction,
        "ground_truth": answer,
        "input": input_text,
        "got_enhanced": got_info is not None
    }
    
    # 如果有GoT信息，添加到结果中
    if got_info:
        result["got_score"] = got_info.get("best_score", 0.0)
        result["got_iterations"] = got_info.get("iterations", 0)
        
    return result


def main(args, LLM):
    input_file = os.path.join(args.data_path, args.d)
    rule_postfix = "no_rule"
    
    # Load dataset
    dataset = load_dataset(input_file, split=args.split)
    
    if args.add_rule:
        rule_postfix = args.rule_path.replace("/", "_").replace(".", "_")
        rule_dataset = utils.load_jsonl(args.rule_path)
        
        # 使用支持GoT的合并函数
        dataset = merge_rule_result_with_got(
            dataset, rule_dataset, args.n, args.filter_empty
        )
        
        # 检查是否是GoT增强的结果
        if any("got_info" in d for d in rule_dataset):
            rule_postfix += "_got_enhanced"
            logger.info("Detected GoT-enhanced rule paths")
            
        if args.use_true:
            rule_postfix = "ground_rule"
        elif args.use_random:
            rule_postfix = "random_rule"

    if args.cot:
        rule_postfix += "_cot"
    if args.explain:
        rule_postfix += "_explain"
    if args.filter_empty:
        rule_postfix += "_filter_empty"
    if args.each_line:
        rule_postfix += "_each_line"
        
    print("Load dataset finished")
    output_dir = os.path.join(
        args.predict_path, args.d, args.model_name, args.split, rule_postfix
    )
    print("Save results to: ", output_dir)
    
    # Predict
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if LLM is not None:
        model = LLM(args)
        
        # 使用GoT增强的提示构建器（如果启用）
        if args.use_got_prompts:
            input_builder = GoTEnhancedPromptBuilder(
                args.prompt_path,
                args.add_rule,
                use_true=args.use_true,
                cot=args.cot,
                explain=args.explain,
                use_random=args.use_random,
                each_line=args.each_line,
                maximun_token=model.maximun_token,
                tokenize=model.tokenize,
            )
        else:
            input_builder = PromptBuilder(
                args.prompt_path,
                args.add_rule,
                use_true=args.use_true,
                cot=args.cot,
                explain=args.explain,
                use_random=args.use_random,
                each_line=args.each_line,
                maximun_token=model.maximun_token,
                tokenize=model.tokenize,
            )
            
        print("Prepare pipeline for inference...")
        model.prepare_for_inference()
    else:
        model = None
        input_builder = PromptBuilder(
            args.prompt_path, args.add_rule, use_true=args.use_true
        )

    # Save args file
    with open(os.path.join(output_dir, "args.txt"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    output_file = os.path.join(output_dir, f"predictions.jsonl")
    fout, processed_list = get_output_file(output_file, force=args.force)

    # 统计GoT增强的样本
    got_enhanced_count = 0
    
    if args.n > 1:
        with Pool(args.n) as p:
            for res in tqdm(
                p.imap(
                    partial(
                        prediction_with_got,
                        processed_list=processed_list,
                        input_builder=input_builder,
                        model=model,
                    ),
                    dataset,
                ),
                total=len(dataset),
            ):
                if res is not None:
                    if res.get("got_enhanced", False):
                        got_enhanced_count += 1
                    if args.debug:
                        print(json.dumps(res))
                    fout.write(json.dumps(res) + "\n")
                    fout.flush()
    else:
        for data in tqdm(dataset):
            res = prediction_with_got(data, processed_list, input_builder, model)
            if res is not None:
                if res.get("got_enhanced", False):
                    got_enhanced_count += 1
                if args.debug:
                    print(json.dumps(res))
                fout.write(json.dumps(res) + "\n")
                fout.flush()
                
    fout.close()
    
    # 打印GoT统计信息
    if got_enhanced_count > 0:
        print(f"\nGoT-enhanced samples: {got_enhanced_count}/{len(dataset)}")
        
        # 计算GoT增强样本的平均分数
        got_scores = []
        with open(output_file, "r") as f:
            for line in f:
                data = json.loads(line)
                if data.get("got_enhanced", False) and "got_score" in data:
                    got_scores.append(data["got_score"])
                    
        if got_scores:
            avg_got_score = sum(got_scores) / len(got_scores)
            print(f"Average GoT score: {avg_got_score:.3f}")

    # 评估结果
    eval_result(output_file)
    
    # 如果有GoT增强的结果，生成额外的分析
    if got_enhanced_count > 0:
        analyze_got_impact(output_file)


def analyze_got_impact(prediction_file):
    """分析GoT增强的影响"""
    got_results = []
    baseline_results = []
    
    with open(prediction_file, "r") as f:
        for line in f:
            data = json.loads(line)
            if data.get("got_enhanced", False):
                got_results.append(data)
            else:
                baseline_results.append(data)
                
    if not got_results:
        return
        
    analysis_file = prediction_file.replace("predictions.jsonl", "got_analysis.txt")
    
    with open(analysis_file, "w") as f:
        f.write("GoT Enhancement Analysis\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total samples: {len(got_results) + len(baseline_results)}\n")
        f.write(f"GoT-enhanced samples: {len(got_results)}\n")
        f.write(f"Baseline samples: {len(baseline_results)}\n\n")
        
        # 计算平均GoT分数和迭代次数
        if got_results:
            avg_score = sum(r.get("got_score", 0) for r in got_results) / len(got_results)
            avg_iterations = sum(r.get("got_iterations", 0) for r in got_results) / len(got_results)
            
            f.write(f"Average GoT confidence score: {avg_score:.3f}\n")
            f.write(f"Average GoT iterations: {avg_iterations:.1f}\n\n")
            
        # 分析高置信度样本
        high_confidence = [r for r in got_results if r.get("got_score", 0) > 0.8]
        if high_confidence:
            f.write(f"High confidence samples (>0.8): {len(high_confidence)}\n")
            f.write("Sample high confidence questions:\n")
            for r in high_confidence[:5]:
                f.write(f"  - {r['question']} (score: {r.get('got_score', 0):.3f})\n")
                
    print(f"\nGoT analysis saved to: {analysis_file}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_path", type=str, default="rmanluo")
    argparser.add_argument("--d", "-d", type=str, default="RoG-webqsp")
    argparser.add_argument("--split", type=str, default="test")
    argparser.add_argument("--predict_path", type=str, default="results/KGQA")
    argparser.add_argument("--model_name", type=str, default="gpt-3.5-turbo")
    argparser.add_argument("--prompt_path", type=str, default="prompts/llama2_predict.txt")
    argparser.add_argument("--add_rule", action="store_true")
    argparser.add_argument("--use_true", action="store_true")
    argparser.add_argument("--cot", action="store_true")
    argparser.add_argument("--explain", action="store_true")
    argparser.add_argument("--use_random", action="store_true")
    argparser.add_argument("--each_line", action="store_true")
    argparser.add_argument("--rule_path", type=str, 
                          default="results/gen_rule_path/webqsp/RoG/test/predictions_3_False.jsonl")
    argparser.add_argument("--force", "-f", action="store_true")
    argparser.add_argument("-n", default=1, type=int, help="number of processes")
    argparser.add_argument("--filter_empty", action="store_true")
    argparser.add_argument("--debug", action="store_true")
    argparser.add_argument("--use_got_prompts", action="store_true", 
                          help="Use GoT-enhanced prompts")

    args, _ = argparser.parse_known_args()
    if args.model_name != "no-llm":
        LLM = get_registed_model(args.model_name)
        LLM.add_args(argparser)
    else:
        LLM = None
    args = argparser.parse_args()

    main(args, LLM)