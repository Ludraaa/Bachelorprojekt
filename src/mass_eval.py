
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType, AutoPeftModelForCausalLM
import os
from transformers import TrainerCallback
import subprocess
import sys
import json

checkpoint_dir = "../Data/Models/MyModels/AllModels/MyWikiSP_ALL/"
model_name = "MyWikiSP_ALL"
mongo_port1=27017
mongo_port2=27018
data_path_1="../Data/Datasets/WikiWebQuestions/"
data_path_2="../Data/Datasets/Qald7/"
eval_mode_1="dev"
eval_mode_2="test"
comparison_path_1="../Data/PredictedResults/local_wikisp-q7_wwq_dev.json"
comparison_path_2="../Data/PredictedResults/local_wikisp-q7_q7_test.json"
dataset_is_wwq_1=True
dataset_is_wwq_2=False
output_path="MyWikiSP_ALL-training_results.csv"

ret0 = None
ret1 = None

def run_eval(model ,data_path, eval_mode,
                  save_path, comparison_path, is_wwq, mongo_port):

    #path to eval venv (training uses a different venv)
    EVAL_PYTHON = "/work/dlclarge2/drayerl-Bachelorprojekt_WS/venv/bin/python"

    #path to the eval script
    EVAL_SCRIPT = "/work/dlclarge2/drayerl-Bachelorprojekt_WS/src/eval.py"

    cmd = [
        EVAL_PYTHON, EVAL_SCRIPT,
        "--checkpoint_path", model,
        "--data_path", data_path,
        "--eval_mode", eval_mode,
        "--dataset_is_wwq", str(is_wwq),
        "--mongo_port", str(mongo_port)
    ]

    if save_path:
        cmd += ["--save_path", save_path]
    if comparison_path:
        cmd += ["--comparison_path", comparison_path]

    print(f"[EvalCallback] Running eval: {data_path}")
    try:
        result = subprocess.run(cmd, check=True, 
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                text=True)

    except subprocess.CalledProcessError as e:
        print("=== Eval subprocess failed ===")
        print("Command:", e.cmd)
        print("Return code:", e.returncode)
        print("STDOUT:\n", e.stdout)
        print("STDERR:\n", e.stderr)
        raise  # optional: re-raise so training still stops

    # The last line is your JSON
    last_line = result.stdout.strip().splitlines()[-1]

    try:
        return json.loads(last_line)

    except json.JSONDecodeError:
        print("[EvalCallback] Failed to parse JSON output from eval.py")
        print(result.stdout)
        raise


checkpoints = []
for name in os.listdir(checkpoint_dir):
    if name.startswith("checkpoint-"):
        try:
            step = int(name.split("-")[1])
            checkpoints.append((step, os.path.join(checkpoint_dir, name)))
        except ValueError:
            continue

# Sort ascending by step
checkpoints.sort(key=lambda x: x[0])
print("Checkpoints to check:\n")
print(checkpoints)

# Iterate in order
for step, path in checkpoints:
    print(f"Processing checkpoint {step} at {path}")
    VENV_PATH = "/work/dlclarge2/drayerl-Bachelorprojekt_WS/venv/bin/python"
    SCRIPT_PATH = "/work/dlclarge2/drayerl-Bachelorprojekt_WS/src/merge.py"

    merge_cmd = [
        VENV_PATH,
        SCRIPT_PATH,
        "--base_model_path", "../Data/Models/Llama-2-7b-hf",
        "--lora_checkpoint_path", path,
        "--model_name", model_name                  
    ]

    print(f"Merging into model using: {path}")

    subprocess.run(merge_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print("Merging complete.")

    merged_path = os.path.join(checkpoint_dir, f"{model_name}{step}")

    print(f"Evaluating using model: {merged_path}")    
    
    # Run eval #1
    ret0 = run_eval(
        merged_path,
        data_path_1,
        eval_mode_1,
        None, #save_path_1,
        comparison_path_1,
        dataset_is_wwq_1,
        mongo_port1
        )
    print(ret0)

    # Run eval #2 (if configured)
    if data_path_2:
        ret1 = run_eval(
            merged_path,
            data_path_2,
            eval_mode_2,
            None, #save_path_2,
            comparison_path_2,
            dataset_is_wwq_2,
            mongo_port2
        )
        print(ret1)

    # Parse eval results
    res1 = ret0 if ret0 else {"acc": None, "f1": None, "local_acc": None}
    res2 = ret1 if ret1 else {"acc": None, "f1": None, "local_acc": None}

    # TSV file path
    csv_path = output_path

    # Ensure header exists
    write_header = not os.path.exists(csv_path)

    tmp_model_name = os.path.basename(merged_path)
    dataset_name_1 = os.path.basename(data_path_1.rstrip("/"))
    dataset_name_2 = os.path.basename(data_path_2.rstrip("/"))

    def pct(x):
        return f"{float(x) * 100:.1f}"

    with open(csv_path, "a") as f:
        if write_header:
            f.write(
                "model,"
                f"{dataset_name_1}.{eval_mode_1}_acc,"
                f"{dataset_name_1}.{eval_mode_1}_f1,"
                f"{dataset_name_1}.{eval_mode_1}_local_acc,"
                f"{dataset_name_2}.{eval_mode_2}_acc,"
                f"{dataset_name_2}.{eval_mode_2}_f1,"
                f"{dataset_name_2}.{eval_mode_2}_local_acc\n"
            )

        f.write(
            f"{tmp_model_name},"
            f"{pct(res1['acc'])},{pct(res1['f1'])},{pct(res1['local_acc'])},"
            f"{pct(res2['acc'])},{pct(res2['f1'])},{pct(res2['local_acc'])}\n"
        )


