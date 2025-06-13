# src/qa_prediction/gen_rule_path.py
# 集成GoT推理 - 修复序列化问题

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

# 导入GoT模块
try:
    from got_reasoning.got_engine import GoTEngine, GoTConfig
    from got_reasoning.minimal_got import integrate_minimal_got
    GOT_AVAILABLE = True
except ImportError:
    GOT_AVAILABLE = False
    print("Warning: GoT modules not found. Running without GoT enhancement.")

N_CPUS = (
    int(os.environ["SLURM_CPUS_PER_TASK"]) if "SLURM_CPUS_PER_TASK" in os.environ else 1
)
PATH_RE = r"<PATH>(.*)<\/PATH>"
INSTRUCTION = """Please generate a valid relation path that can be helpful for answering the following question: """


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


def generate_seq(
    model, input_text, tokenizer, num_beam=3, do_sample=False, max_new_tokens=100
):
    """Generate sequences using beam search"""
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda")
    
    output = model.generate(
        input_ids=input_ids,
        num_beams=num_beam,
        num_return_sequences=num_beam,
        early_stopping=False,
        do_sample=do_sample,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=max_new_tokens,
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


def gen_prediction(args):
    """主预测函数 - 支持GoT增强"""
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    if args.lora or os.path.exists(args.model_path + "/adapter_config.json"):
        print("Load LORA model")
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

    input_file = os.path.join(args.data_path, args.d)
    output_dir = os.path.join(args.output_path, args.d, args.model_name, args.split)
    print("Save results to: ", output_dir)

    # Load dataset
    dataset = load_dataset(input_file, split=args.split)

    # Load prompt template
    prompter = utils.InstructFormater(args.prompt_path)

    def prepare_dataset(sample):
        # Prepare input prompt
        sample["text"] = prompter.format(
            instruction=INSTRUCTION, message=sample["question"]
        )
        # Find ground-truth paths for each Q-P pair
        # 不要将graph对象保存到dataset中，避免序列化错误
        graph = utils.build_graph(sample["graph"])
        paths = utils.get_truth_paths(sample["q_entity"], sample["a_entity"], graph)
        ground_paths = set()
        for path in paths:
            ground_paths.add(tuple([p[1] for p in path]))
        sample["ground_paths"] = list(ground_paths)
        # 不保存graph对象
        # sample["kg_graph"] = graph  # 删除这行
        return sample

    dataset = dataset.map(prepare_dataset, num_proc=N_CPUS)

    # Predict
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 配置GoT（如果启用）
    got_engine = None
    llm_wrapper = None
    if args.use_got and GOT_AVAILABLE:
        # 使用保守配置
        got_config = GoTConfig(
            max_iterations=args.got_iterations,
            beam_width=args.got_beam_width,
            score_threshold=args.got_score_threshold,
            enable_feedback=args.got_enable_feedback,
            use_graph_attention=args.got_use_attention,
            aggregation_strategy=args.got_aggregation,
            # 保守配置
            preserve_original=True,  # 总是保留原始路径
            max_enhanced_paths=3,    # 限制增强数量
            enable_minimal_mode=args.got_minimal_mode  # 使用最小化模式
        )
        
        # 创建LLM包装器
        class LLMWrapper:
            def __init__(self, model, tokenizer):
                self.model = model
                self.tokenizer = tokenizer
                
            def generate_sentence(self, prompt):
                inputs = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_new_tokens=200,
                        temperature=0.7,
                        do_sample=True
                    )
                return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        llm_wrapper = LLMWrapper(model, tokenizer)
    
    # 设置输出文件名
    suffix = ""
    if args.use_got and GOT_AVAILABLE:
        suffix = "_got"
    elif args.use_minimal_got and GOT_AVAILABLE:
        suffix = "_minimal"
        
    prediction_file = os.path.join(
        output_dir, f"predictions_{args.n_beam}_{args.do_sample}{suffix}.jsonl"
    )
    f, processed_results = get_output_file(prediction_file, force=args.force)
    
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
        
        # 步骤2：应用GoT增强（如果启用）
        got_info = None
        if args.use_got and initial_paths and GOT_AVAILABLE:
            # 重新构建graph（因为没有保存在dataset中）
            kg_graph = utils.build_graph(data["graph"])
            got_engine = GoTEngine(got_config, llm_wrapper, kg_graph)
            
            # 执行GoT推理
            if got_config.enable_minimal_mode:
                # 同步调用
                got_result = got_engine._minimal_reasoning(question, initial_paths)
            else:
                # 异步调用
                got_result = asyncio.run(got_engine.reason(question, initial_paths))
            
            # 提取增强后的路径
            rel_paths = got_result.get("reasoning_paths", initial_paths)
            
            # 保存GoT信息
            got_info = {
                "thought_graph_size": got_result.get("total_thoughts", 0),
                "iterations": got_result.get("iterations", 1),
                "best_score": got_result.get("score", 0),
                "initial_paths": initial_paths,
                "enhanced_paths": rel_paths,
                "minimal_mode": got_result.get("minimal_mode", False)
            }
        elif args.use_minimal_got and initial_paths and GOT_AVAILABLE:
            # 使用独立的最小化GoT
            rel_paths = integrate_minimal_got(question, initial_paths)
            got_info = {
                "minimal_got": True,
                "initial_count": len(initial_paths),
                "enhanced_count": len(rel_paths)
            }
        else:
            rel_paths = initial_paths
            
        if args.debug:
            print("ID: ", qid)
            print("Question: ", question)
            print("Initial paths: ", initial_paths)
            if got_info:
                print("Enhanced paths: ", rel_paths)
                
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
    return prediction_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="rmanluo")
    parser.add_argument("--d", "-d", type=str, default="RoG-webqsp")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_path", type=str, default="results/gen_rule_path")
    parser.add_argument("--model_name", type=str, default="Llama-2-7b-hf")
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--prompt_path", type=str, default="prompts/llama2.txt")
    parser.add_argument("--rel_dict", nargs="+", default=["datasets/KG/fbnet/relations.dict"])
    parser.add_argument("--force", "-f", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--n_beam", type=int, default=1)
    parser.add_argument("--do_sample", action="store_true")
    
    # GoT相关参数
    parser.add_argument("--use_got", action="store_true", help="Enable Graph of Thoughts reasoning")
    parser.add_argument("--use_minimal_got", action="store_true", help="Use minimal GoT (safest option)")
    parser.add_argument("--got_minimal_mode", action="store_true", default=True, help="Use minimal mode in GoT engine")
    parser.add_argument("--got_iterations", type=int, default=1, help="GoT iterations")
    parser.add_argument("--got_beam_width", type=int, default=5, help="GoT beam width")
    parser.add_argument("--got_score_threshold", type=float, default=0.8, help="GoT score threshold")
    parser.add_argument("--got_enable_feedback", action="store_true", default=False, help="Enable feedback loop")
    parser.add_argument("--got_use_attention", action="store_true", default=False, help="Use graph attention")
    parser.add_argument("--got_aggregation", type=str, default="none", 
                       choices=["adaptive", "greedy", "exhaustive", "none"])

    args = parser.parse_args()
    
    # 检查GoT是否可用
    if (args.use_got or args.use_minimal_got) and not GOT_AVAILABLE:
        print("Warning: GoT requested but modules not available. Running without GoT.")
        args.use_got = False
        args.use_minimal_got = False
    
    gen_path = gen_prediction(args)