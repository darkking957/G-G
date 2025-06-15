# src/qa_prediction/gen_rule_path.py
# 修改版本：启用完整的GoT推理功能

import json
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

import argparse
import utils
from datasets import load_dataset
import datasets
datasets.disable_progress_bar()
from tqdm import tqdm
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM
import torch
import re
import asyncio
import logging

# 导入GoT模块
from got_reasoning.got_engine import GoTEngine, GoTConfig
from got_reasoning.thought_graph import ThoughtGraph
from got_reasoning.validate_plans import PlanValidator
from got_reasoning.evaluate_plans import SemanticEvaluator
from got_reasoning.aggregate_plans import ThoughtAggregator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

N_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
PATH_RE = r"<PATH>(.*)<\/PATH>"
INSTRUCTION = """Please generate a valid relation path that can be helpful for answering the following question: """


class LLMWrapper:
    """LLM包装器，提供给GoT组件使用"""
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def generate_sentence(self, prompt, temperature=0.7):
        """生成单个句子"""
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=200,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    def tokenize(self, text):
        """分词"""
        return self.tokenizer.tokenize(text)


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


def parse_prediction(prediction):
    """Parse a list of predictions to a list of rules"""
    results = []
    for p in prediction:
        path = re.search(PATH_RE, p)
        if path is None:
            continue
        path = path.group(1)
        path = path.split("<SEP>")
        if len(path) == 0:
            continue
        rules = []
        for rel in path:
            rel = rel.strip()
            if rel == "":
                continue
            rules.append(rel)
        results.append(rules)
    return results


def generate_seq(model, input_text, tokenizer, num_beam=3, do_sample=False, max_new_tokens=100):
    """Generate sequences using beam search"""
    #input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda")
    inputs = tokenizer(input_text, return_tensors="pt", padding=True).to("cuda")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,  # 添加这一行
        num_beams=num_beam,
        num_return_sequences=num_beam,
        early_stopping=False,
        do_sample=do_sample,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id  # 使用pad_token_id而不是eos_token_id
    )
    
    prediction = tokenizer.batch_decode(
        output.sequences[:, input_ids.shape[1] :], skip_special_tokens=True
    )
    prediction = [p.strip() for p in prediction]

    if num_beam > 1:
        scores = output.sequences_scores.tolist()
        norm_scores = torch.softmax(output.sequences_scores, dim=0).tolist()
    else:
        scores = [1]
        norm_scores = [1]

    return {"paths": prediction, "scores": scores, "norm_scores": norm_scores}


async def apply_got_reasoning(question, initial_paths, kg_graph, llm_wrapper, config):
    """应用完整的GoT推理"""
    # 创建GoT引擎
    engine = GoTEngine(config, llm_wrapper, kg_graph)
    
    # 执行推理
    result = await engine.reason(question, initial_paths)
    
    # 生成解释
    explanation = await engine.explain_reasoning(result)
    result["explanation"] = explanation
    
    return result


def gen_prediction(args):
    """主预测函数 - 支持完整GoT"""
    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    elif tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.unk_token
    if args.lora or os.path.exists(args.model_path + "/adapter_config.json"):
        logger.info("Loading LORA model")
        model = AutoPeftModelForCausalLM.from_pretrained(
            args.model_path, device_map="auto", torch_dtype=torch.bfloat16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            use_auth_token=True,
        )

    # 创建LLM包装器
    llm_wrapper = LLMWrapper(model, tokenizer)

    input_file = os.path.join(args.data_path, args.d)
    output_dir = os.path.join(args.output_path, args.d, args.model_name, args.split)
    logger.info(f"Save results to: {output_dir}")

    # Load dataset
    dataset = load_dataset(input_file, split=args.split)

    # Load prompt template
    prompter = utils.InstructFormater(args.prompt_path)

    def prepare_dataset(sample):
        # Prepare input prompt
        sample["text"] = prompter.format(
            instruction=INSTRUCTION, message=sample["question"]
        )
        # Find ground-truth paths
        graph = utils.build_graph(sample["graph"])
        paths = utils.get_truth_paths(sample["q_entity"], sample["a_entity"], graph)
        ground_paths = set()
        for path in paths:
            ground_paths.add(tuple([p[1] for p in path]))
        sample["ground_paths"] = list(ground_paths)
        return sample

    dataset = dataset.map(prepare_dataset, num_proc=N_CPUS)

    # Predict
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 配置GoT
    if args.use_got:
        got_config = GoTConfig(
            max_iterations=args.got_iterations,
            beam_width=args.got_beam_width,
            score_threshold=args.got_score_threshold,
            enable_feedback=args.got_enable_feedback,
            use_graph_attention=args.got_use_attention,
            aggregation_strategy=args.got_aggregation,
            validation_mode="relaxed",
            # 关键：禁用最小化模式
            enable_minimal_mode=False,
            preserve_original=False
        )
        logger.info(f"GoT Configuration: {got_config}")
    
    # 设置输出文件名
    suffix = "_got" if args.use_got else ""
    prediction_file = os.path.join(
        output_dir, f"predictions_{args.n_beam}_{args.do_sample}{suffix}.jsonl"
    )
    f, processed_results = get_output_file(prediction_file, force=args.force)
    
    # 处理每个样本
    for data in tqdm(dataset):
        question = data["question"]
        input_text = data["text"]
        qid = data["id"]
        
        if qid in processed_results:
            continue
            
        # 步骤1：生成初始路径候选
        raw_output = generate_seq(
            model,
            input_text,
            tokenizer,
            max_new_tokens=args.max_new_tokens,
            num_beam=args.n_beam,
            do_sample=args.do_sample,
        )
        initial_paths = parse_prediction(raw_output["paths"])
        
        # 步骤2：应用GoT增强
        got_info = None
        if args.use_got and initial_paths:
            # 构建知识图谱
            kg_graph = utils.build_graph(data["graph"])
            
            # 执行完整的GoT推理
            logger.info(f"Applying GoT reasoning for question: {question[:50]}...")
            got_result = asyncio.run(
                apply_got_reasoning(question, initial_paths, kg_graph, llm_wrapper, got_config)
            )
            
            # 提取结果
            rel_paths = got_result.get("reasoning_paths", initial_paths)
            
            # 保存GoT信息
            got_info = {
                "best_score": got_result.get("score", 0),
                "iterations": got_result.get("iterations", 0),
                "total_thoughts": got_result.get("total_thoughts", 0),
                "statistics": got_result.get("statistics", {}),
                "explanation": got_result.get("explanation", ""),
                "initial_paths": initial_paths,
                "enhanced_paths": rel_paths,
                "aggregation_performed": got_result.get("statistics", {}).get("aggregated_thoughts", 0) > 0,
                "feedback_applied": got_result.get("statistics", {}).get("improved_thoughts", 0) > 0
            }
        else:
            rel_paths = initial_paths
            
        if args.debug:
            print(f"\nID: {qid}")
            print(f"Question: {question}")
            print(f"Initial paths ({len(initial_paths)}): {initial_paths[:3]}")
            if got_info:
                print(f"Enhanced paths ({len(rel_paths)}): {rel_paths[:3]}")
                print(f"Best score: {got_info['best_score']:.3f}")
                print(f"Statistics: {got_info['statistics']}")
                
        # Save results
        data_to_save = {
            "id": qid,
            "question": question,
            "prediction": rel_paths,
            "ground_paths": data["ground_paths"],
            "input": input_text,
            "raw_output": raw_output,
        }
        
        if got_info:
            data_to_save["got_info"] = got_info
            
        f.write(json.dumps(data_to_save) + "\n")
        f.flush()
        
    f.close()
    
    # 打印总结统计
    if args.use_got:
        logger.info("\n=== GoT Reasoning Summary ===")
        logger.info(f"Output file: {prediction_file}")
        logger.info("GoT enhancement completed successfully!")
        
    return prediction_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="rmanluo")
    parser.add_argument("--d", "-d", type=str, default="RoG-webqsp")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_path", type=str, default="results/gen_rule_path")
    parser.add_argument("--model_name", type=str, default="RoG")
    parser.add_argument("--model_path", type=str, default="rmanluo/RoG")
    parser.add_argument("--prompt_path", type=str, default="prompts/llama2.txt")
    parser.add_argument("--rel_dict", nargs="+", default=["datasets/KG/fbnet/relations.dict"])
    parser.add_argument("--force", "-f", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--n_beam", type=int, default=3)
    parser.add_argument("--do_sample", action="store_true")
    
    # GoT相关参数
    parser.add_argument("--use_got", action="store_true", help="Enable Graph of Thoughts reasoning")
    parser.add_argument("--got_iterations", type=int, default=3, help="Number of GoT iterations")
    parser.add_argument("--got_beam_width", type=int, default=10, help="Beam width for thought selection")
    parser.add_argument("--got_score_threshold", type=float, default=0.5, help="Score threshold")
    parser.add_argument("--got_enable_feedback", action="store_true", default=True, help="Enable feedback loop")
    parser.add_argument("--got_use_attention", action="store_true", default=True, help="Use graph attention")
    parser.add_argument("--got_aggregation", type=str, default="adaptive", 
                       choices=["adaptive", "greedy", "exhaustive", "none"],
                       help="Aggregation strategy")

    args = parser.parse_args()
    
    gen_path = gen_prediction(args)