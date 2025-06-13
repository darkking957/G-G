import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from utils import *
import datasets
from qa_prediction.build_qa_input import PromptBuilder
from transformers import AutoTokenizer
from llms.language_models import ChatGPT
from collections import namedtuple
import multiprocessing as mp
from tqdm import tqdm

N_CPUS = int(os.environ['SLURM_CPUS_PER_TASK']) if 'SLURM_CPUS_PER_TASK' in os.environ else 1

save_dir = "datasets/joint_training/ExplainQAData"
split="train"
model_max_length = 1024
#data_list = ['RoG-webqsp', 'RoG-cwq']
data_list = ['vul']
data_path = "rmanluo"

Config = namedtuple("Config", "retry model_name")
config = Config(retry=3, model_name = "deepseek-chat")
model = ChatGPT(config)
gpt_prompt_path = "prompts/general_prompt.txt"
model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
    )

few_shot_examples = """## Input:
        Based on the reasoning paths, please answer the given question and explain why.

        Reasoning Paths:
        CVE-2024-52823 -> has_severity -> MEDIUM
        CVE-2024-6582 -> has_severity -> MEDIUM
        CVE-2024-46903 -> has_severity -> MEDIUM

        Question:
        How can the exploitation of multiple medium-severity CVEs (e.g., CVE-2024-52823, CVE-2024-6582, CVE-2024-46903) be chained together to escalate an attacker's privileges or achieve a more severe impact than exploiting any single vulnerability in isolation?

        ## Output:
        Answer:
        MEDIUM

        Explanation:
        1. The reasoning paths show that multiple CVEs (CVE-2024-52823, CVE-2024-6582, and CVE-2024-46903) all have the same severity level: MEDIUM.
        2. This is directly provided in the reasoning paths where each CVE is linked to its severity level through the "has_severity" relationship.
        3. The question asks about how these medium-severity CVEs can be chained together, confirming that their common attribute is their MEDIUM severity level.

        Therefore, based on the provided reasoning paths, the answer is MEDIUM, as all the mentioned CVEs share this severity level which makes them candidates for vulnerability chaining.

        ## Input:
        Based on the reasoning paths, please answer the given question and explain why.

        Reasoning Paths:
        CVE-2024-6367 -> affects -> LabVantage:LIMS
        LabVantage:LIMS -> vulnerable_to -> CVE-2024-6367
        CVE-2024-6367 -> has_weakness -> CWE-CWE-79
        CWE-CWE-79 -> manifested_in -> CVE-2024-6367

        Question:
        How does the weakness CWE-79 (Cross-site Scripting) manifest in the vulnerability CVE-2024-6367 affecting LabVantage:LIMS, and what potential attack techniques could exploit this relationship?

        ## Output:
        Answer:
        LabVantage:LIMS

        Explanation:
        1. The reasoning paths show that CVE-2024-6367 affects LabVantage:LIMS, as indicated by the direct relationship "CVE-2024-6367 -> affects -> LabVantage:LIMS".
        2. This is reinforced by the inverse relationship "LabVantage:LIMS -> vulnerable_to -> CVE-2024-6367".
        3. Additionally, the paths indicate that CVE-2024-6367 has the weakness CWE-CWE-79 (Cross-site Scripting), which is manifested in this vulnerability.
        4. The question asks about how CWE-79 manifests in CVE-2024-6367 affecting LabVantage:LIMS, confirming that LabVantage:LIMS is the affected product.

        Therefore, based on the provided reasoning paths, LabVantage:LIMS is the product affected by the vulnerability CVE-2024-6367 which manifests the CWE-79 weakness.

        ## Input:
        Based on the reasoning paths, please answer the given question and explain why.

        Reasoning Paths:
        CVE-2024-32808 -> affects -> Metagauss:ProfileGrid
        Metagauss:ProfileGrid -> vulnerable_to -> CVE-2024-32808
        CVE-2024-31362 -> affects -> Metagauss:ProfileGrid
        Metagauss:ProfileGrid -> vulnerable_to -> CVE-2024-31362
        CVE-2024-30241 -> affects -> Metagauss:ProfileGrid
        CVE-2024-32774 -> affects -> Metagauss:ProfileGrid

        Question:
        What attack techniques could be leveraged to exploit the multiple CVEs (CVE-2024-32808, CVE-2024-31362, CVE-2024-30241, CVE-2024-32774) in Metagauss:ProfileGrid, and how might an attacker chain these vulnerabilities to escalate privileges or achieve remote code execution?

        ## Output:
        Answer:
        Metagauss:ProfileGrid

        Explanation:
        1. The reasoning paths clearly show that multiple CVEs affect the same product, Metagauss:ProfileGrid.
        2. Specifically, CVE-2024-32808, CVE-2024-31362, CVE-2024-30241, and CVE-2024-32774 all have the relationship "affects -> Metagauss:ProfileGrid".
        3. This is reinforced by the inverse relationships showing that "Metagauss:ProfileGrid -> vulnerable_to" these various CVEs.
        4. The question asks about attack techniques for exploiting these multiple CVEs in Metagauss:ProfileGrid, confirming that Metagauss:ProfileGrid is the affected product.

        Therefore, based on the provided reasoning paths, Metagauss:ProfileGrid is the product affected by multiple vulnerabilities that could potentially be chained together in an attack.

        ## Input:
        Based on the reasoning paths, please answer the given question and explain why.

        Reasoning Paths:
        N -> exploits -> CVE-2024-38743
        CVE-2024-38743 -> exploited_by -> N
        N -> exploits -> CVE-2024-4885
        N -> exploits -> CVE-2024-6582
        N -> exploits -> CVE-2024-46903

        Question:
        What is the significance of an attacker (N) exploiting multiple CVEs (CVE-2024-38743, CVE-2024-4885, CVE-2024-6582, CVE-2024-46903), and how might this pattern indicate a broader attack strategy or operational behavior in the cybersecurity landscape?

        ## Output:
        Answer:
        N

        Explanation:
        1. The reasoning paths show that entity N exploits multiple CVEs, specifically CVE-2024-38743, CVE-2024-4885, CVE-2024-6582, and CVE-2024-46903.
        2. This is demonstrated through multiple "N -> exploits -> [CVE]" relationships.
        3. The inverse relationship "CVE-2024-38743 -> exploited_by -> N" further confirms this connection.
        4. The question asks about the significance of attacker N exploiting these multiple CVEs, confirming that N is the entity responsible for the exploitation.

        Therefore, based on the provided reasoning paths, N is the attacker that exploits the multiple CVEs mentioned in the question.

        ## Input:
        Based on the reasoning paths, please answer the given question and explain why.

        Reasoning Paths:
        CVE-2024-33699 -> has_weakness -> CWE-CWE-620
        CWE-CWE-620 -> manifested_in -> CVE-2024-33699
        CVE-2024-13373 -> has_weakness -> CWE-CWE-620
        CVE-2024-12824 -> has_weakness -> CWE-CWE-620
        CVE-2024-20419 -> has_weakness -> CWE-CWE-620

        Question:
        What is the common weakness (CWE) shared by CVEs 2024-33699, 2024-13373, 2024-12824, and 2024-20419, and how does this weakness typically manifest in these vulnerabilities?

        ## Output:
        Answer:
        CWE-CWE-620

        Explanation:
        1. The reasoning paths show that multiple CVEs (CVE-2024-33699, CVE-2024-13373, CVE-2024-12824, and CVE-2024-20419) all share the same weakness: CWE-CWE-620.
        2. This is demonstrated by the "has_weakness" relationship connecting each CVE to CWE-CWE-620.
        3. The inverse relationship "CWE-CWE-620 -> manifested_in -> CVE-2024-33699" further confirms this connection.
        4. The question asks about the common weakness shared by these CVEs, confirming that CWE-CWE-620 is the common weakness.

        Therefore, based on the provided reasoning paths, CWE-CWE-620 is the common weakness shared by the CVEs mentioned in the question.

        ## Input:
"""


# Load prompt template
gpt_builder = PromptBuilder(
        gpt_prompt_path,
        add_rule = True,
        use_true= True,
        explain=True,
        maximun_token=model_max_length,
        tokenize=model.tokenize,
    )

def formatting_prompts_func(example):
    output_label = "\n".join(example['answer'])
     # Find ground-truth paths for each Q-P pair
    graph = build_graph(example["graph"])
    paths = get_truth_paths(example["q_entity"], example["a_entity"], graph)
    ground_paths = set()
    for path in paths:
        ground_paths.add(tuple([p[1] for p in path]))  # extract relation path
    if len(ground_paths) == 0:
        return None
    example["ground_paths"] = list(ground_paths)
    input_text = gpt_builder.process_input(example)
    # Filter empty
    if "Reasoning Paths:\n\n\n" in input_text:
        return None
    few_shot_input = few_shot_examples + input_text
    prediction = model.generate_sentence(few_shot_input)
    if prediction is not None:
        output_text = "[INST] <<SYS>>\n<</SYS>>\n" + input_text + ' [/INST] ' + prediction + tokenizer.eos_token
        return {"text": output_text}
    else:
        return None

for data_name in data_list:
    #nput_file = os.path.join(data_path, data_name)
    data_files = {"train": "datasets/train.parquet", "valid":"datasets/valid.parquet","test": "datasets/test.parquet"}
    train_dataset = datasets.load_dataset("parquet",data_files=data_files,split="train")
    train_dataset = train_dataset.shuffle().select(range(700))
    save_path = os.path.join(save_dir, data_name, data_name + "_train_1000.jsonl")
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, "w") as f:
        print("Processing {}...".format(data_name))
        print("Number of process: {}".format(N_CPUS))
        with mp.Pool(N_CPUS) as pool:
            for example in tqdm(pool.imap_unordered(formatting_prompts_func, train_dataset), total=len(train_dataset)):
                if example is not None:
                    f.write(json.dumps(example) + "\n")
    # train_dataset = train_dataset.map(
    #     formatting_prompts_func,
    #     remove_columns=train_dataset.column_names,
    #     num_proc=N_CPUS,
    # )
    # train_dataset.to_json(save_path, orient="records", lines=True)
