import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType, AutoPeftModelForCausalLM
import os
from transformers import TrainerCallback
import subprocess
import sys
import json

#gets the latest checkpoint in the specified directory
def get_latest_checkpoint(dir):
    latest = None
    latest_step = -1

    for d in os.listdir(dir):
        if d.startswith("checkpoint-"):
            try:
                step = int(d.split("-")[1])
            except ValueError:
                continue

            if step > latest_step:
                latest_step = step
                latest = d

    return os.path.join(dir, latest) if latest else None



class DualEvalCallback(TrainerCallback):
    def __init__(
        self,
        checkpoint_dir,
        base_model_path,
        model_name,
        output_path,
        # First evaluation config
        data_path_1,
        eval_mode_1,
        save_path_1=None,
        comparison_path_1=None,

        # Second evaluation config
        data_path_2=None,
        eval_mode_2=None,
        save_path_2=None,
        comparison_path_2=None,

        # Shared settings
        mongo_port1=27017,
        mongo_port2=27018,

        eval_gpu
    ):
        self.checkpoint_dir = checkpoint_dir
        self.base_model_path = base_model_path
        self.model_name = model_name
        self.output_path = output_path

        # Eval 1
        self.data_path_1 = data_path_1
        self.eval_mode_1 = eval_mode_1
        self.save_path_1 = save_path_1
        self.comparison_path_1 = comparison_path_1
        self.dataset_is_wwq_1 = ("wikiwebquestions" in data_path_1.lower()) or ("wwq" in data_path_1.lower())

        # Eval 2
        self.data_path_2 = data_path_2
        self.eval_mode_2 = eval_mode_2
        self.save_path_2 = save_path_2
        self.comparison_path_2 = comparison_path_2
        self.dataset_is_wwq_2 = (
            ("wikiwebquestions" in data_path_2.lower()) or ("wwq" in data_path_2.lower())
            if data_path_2 is not None else False
        )

        self.mongo_port1 = mongo_port1
        self.mongo_port2 = mongo_port2
            
        #used to store results temporarily
        self.ret0 = None
        self.ret1 = None

        self.gpu = eval_gpu

    def _run_eval(self, model ,data_path, eval_mode,
                  save_path, comparison_path, is_wwq, mongo_port, process_index):

        #path to eval venv (training uses a different venv)
        EVAL_PYTHON = os.environ.get("VENV_EVAL_PATH", "/opt/venv_eval/") + "bin/python"

        #path to the eval script
        EVAL_SCRIPT = "src/eval.py"

        cmd = [
            EVAL_PYTHON, EVAL_SCRIPT,
            "--checkpoint_dir", model,
            "--data_dir", data_path,
            "--eval_mode", eval_mode,
            "--get_current_results", str(is_wwq),
            "--mongo_port", str(mongo_port)
        ]

        if save_path:
            cmd += ["--save_path", save_path]
        if comparison_path:
            cmd += ["--comparison_path", comparison_path]
        
        # === ONLY CHANGE GPU FOR THIS SUBPROCESS ===
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu)

        print(f"[EvalCallback] Running eval: {data_path}", flush=True)
        try:
            result = subprocess.run(cmd, check=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        except subprocess.CalledProcessError as e:
            print("=== Eval subprocess failed ===")
            print("Command:", e.cmd)
            print("Return code:", e.returncode)
            print("STDOUT:\n", e.stdout)
            print("STDERR:\n", e.stderr)
            raise

        # The last line is your JSON
        last_line = result.stdout.strip().splitlines()[-1]
        
        try:
            if process_index == 0:
                self.ret0 = json.loads(last_line)
            elif process_index == 1:
                self.ret1 = json.loads(last_line)
        
        except json.JSONDecodeError:
            print("[EvalCallback] Failed to parse JSON output from eval.py")
            print(result.stdout)
            raise
        

    def on_save(self, args, state, control, **kwargs):

        ckpt_path = get_latest_checkpoint(self.checkpoint_dir)
            
        VENV_PATH = os.environ.get("VENV_TRAIN_PATH", "/opt/venv_train/") + "bin/python"
        SCRIPT_PATH = "src/merge.py"

        merge_cmd = [
            VENV_PATH,
            SCRIPT_PATH,
            "--base_model_path", self.base_model_path,         # you MUST store this in your callback init
            "--lora_checkpoint_path", ckpt_path,
            "--model_name", self.model_name                    # also store this in __init__
        ]
        
        print(f"Merging into model using: {ckpt_path}", flush=True)

         # === ONLY CHANGE GPU FOR THIS SUBPROCESS ===
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        
        try:
            subprocess.run(merge_cmd, check=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except subprocess.CalledProcessError as e:
            print("===Merge subprocess failed===")
            print("Command:", e.cmd)
            print("Return code:", e.returncode)
            print("STDOUT:\n", e.stdout)
            print("STDERR:\n", e.stderr)
            raise

        print("Merging complete.", flush=True)

        ckpt = ckpt_path.split('-')[-1]
        merged_path = os.path.join(self.checkpoint_dir, f"{self.model_name}{ckpt}")
        
        print(f"Evaluating using model: {merged_path}", flush=True)

        # Run eval #1
        self._run_eval(
            merged_path,
            self.data_path_1,
            self.eval_mode_1,
            self.save_path_1,
            self.comparison_path_1,
            self.dataset_is_wwq_1,
            self.mongo_port1,
            0
        )
        print(self.ret0)

        # Run eval #2 (if configured)
        if self.data_path_2:
            self._run_eval(
                merged_path,
                self.data_path_2,
                self.eval_mode_2,
                self.save_path_2,
                self.comparison_path_2,
                self.dataset_is_wwq_2,
                self.mongo_port2,
                1
            )
        print(self.ret1)
        
        # Parse eval results
        res1 = self.ret0 if self.ret0 else {"acc": None, "f1": None, "local_acc": None}
        res2 = self.ret1 if self.ret1 else {"acc": None, "f1": None, "local_acc": None}

        # TSV file path
        csv_path = self.output_path

        # Ensure header exists
        write_header = not os.path.exists(csv_path)
        
        model_name = os.path.basename(merged_path)
        dataset_name_1 = os.path.basename(self.data_path_1.rstrip("/"))
        dataset_name_2 = ""

        if self.data_path_2:
            dataset_name_2 = os.path.basename(self.data_path_2.rstrip("/"))
        else:
            dataset_name_2 = "None"
        

        def pct(x):
            if x:
                return f"{float(x) * 100:.1f}"
            elif x == -1:
                return "None"
            else:
                return "None"

        with open(csv_path, "a") as f:
            if write_header:
                f.write(
                    "model,"
                    f"{dataset_name_1}.{self.eval_mode_1}_acc,"
                    f"{dataset_name_1}.{self.eval_mode_1}_f1,"
                    f"{dataset_name_1}.{self.eval_mode_1}_local_acc,"
                    f"{dataset_name_2}.{self.eval_mode_2}_acc,"
                    f"{dataset_name_2}.{self.eval_mode_2}_f1,"
                    f"{dataset_name_2}.{self.eval_mode_2}_local_acc\n"
                )

            f.write(
                f"{model_name},"
                f"{pct(res1['acc'])},{pct(res1['f1'])},{pct(res1['local_acc'])},"
                f"{pct(res2['acc'])},{pct(res2['f1'])},{pct(res2['local_acc'])}\n"
            )


def main():

    import argparse
    import os
    
    parser = argparse.ArgumentParser(description= "Trains a model on one or multiple datasets. Uses a custom callback for evaluation using eval.py during training.")

    parser.add_argument(
        "--checkpoint_dir_path",
        type=str,
        required=True,
        help="Path to the folder, where the checkpoints during training will be saved."
    )

    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        help="Space seperated list of dataset train sets to load (JSON)"
    )

    parser.add_argument(
        "--scalings",
        type=float,
        nargs="+",
        required=True,
        help="Space seperated list of up-sampling (or down-sampling) multipliers for each dataset."
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="The name of the model."
    )

    parser.add_argument(
       "--base_model_path",
       type=str,
       required=True,
       help="Path to the folder containing the base model to be trained on."
    )

    parser.add_argument(
       "--acc_steps",
       type=int,
       required=True,
       help="Gradient accumulation step size of the hf training parameters class."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="How many epochs to train for."
    )

    parser.add_argument(
        "--eval_steps_per_epoch",
        type=int,
        required=True,
        help="How many times per epoch the eval callback should be called. Setting this to 4 would evaluate the trained models every 0.25 epochs of training."
    )
    
    parser.add_argument(
        "--eval_callback",
        type=bool,
        required=True,
        help="Whether to use the eval callback or train without it."
    )
    
    parser.add_argument(
        "--callback_output_path",
        type=str,
        help="Path to a csv file for the training results to be written."
    )

    parser.add_argument(
        "--callback_data_path",
        type=str,
        help="Path to the directory containing dev and test set of the dataset to be evaluated during the callback."
    )

    parser.add_argument(
        "--callback_eval_mode",
        type=str,
        help="Whether to evaluate on the dev or test set."
    )

    parser.add_argument(
        "--callback_comparison_path",
        type=str,
        help="Path to a saved predictions file that is to be compared to during callback."
    )

    parser.add_argument(
        "--eval_port",
        type=str,
        help="port to be used for the first eval mongodb."
    )

    parser.add_argument(
        "--callback_data_path2",
        type=str,
        help="Path to the directory containing dev and test set of the dataset to be evaluated during the callback."
    )

    parser.add_argument(
        "--callback_eval_mode2",
        type=str,
        help="Whether to evaluate on the dev or test set."
    )

    parser.add_argument(
        "--callback_comparison_path2",
        type=str,
        help="Path to a saved predictions file that is to be compared to during callback."
    )

    parser.add_argument(
        "--eval_port2",
        type=str,
        help="port to be used for the second eval mongodb."
    )
    
    parser.add_argument(
        "--train_gpu",
        type=int,
        default="0",
    )
    parser.add_argument(
        "--eval_gpu",
        type=int,
        default="1",
    )

    args = parser.parse_args()
    
    if len(args.datasets) != len(args.scalings):
        parser.error("Length of --datasets and --scalings must match")
    
    # check for missing arguments when eval callback is true
    if args.eval_callback:
        missing = []
        if args.callback_output_path is None:
            missing.append("--callback_output_path")
        if args.callback_data_path is None:
            missing.append("--callback_data_path")
        if args.callback_eval_mode is None:
            missing.append("--callback_eval_mode")

        if missing:
            parser.error(
                "Missing required arguments when --eval_callback is set: "
                + ", ".join(missing)
            )

    #force training to use specified gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.train_gpu)


    checkpoint_dir = args.checkpoint_dir_path
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_name = args.model_name

    from datasets import load_dataset, concatenate_datasets
    
    #assemble the datasets into one

    loaded_datasets = []

    for ds, factor in zip(args.datasets, args.scalings):
        print(f"Loading dataset: {ds}")
        factor = float(factor)
        ds_original = None
        if ds.lower() == "alpaca":
            ds_original = load_dataset("tatsu-lab/alpaca")["train"]
        else:
            ds_original = load_dataset("json", data_files=ds)["train"]

        if factor >= 1:
            factor = int(round(factor))
            print(f"Upsampling {len(ds_original)} samples by factor {factor} => {int(len(ds_original) * factor)}")
            loaded_datasets.append(concatenate_datasets([ds_original] * factor))
        elif factor < 1:
            print(f"Downsampling {len(ds_original)} samples by factor {factor} => {int(len(ds_original) * factor)}")
            desired_size = int(len(ds_original) * factor)
            loaded_datasets.append(ds_original.shuffle(seed=42).select(range(desired_size)))

    dataset = None

    if len(loaded_datasets) > 1:
        dataset = concatenate_datasets(loaded_datasets)
    else:
        dataset = loaded_datasets[0]

    #shuffle the dataset
    dataset = dataset.shuffle(seed=42)

    print("Combined total size:", len(dataset))


    from transformers import AutoTokenizer


    model_path = args.base_model_path
    tokenizer_path = model_path #always take the base tokenizer, checkpoints dont contain one

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    def format_example(example):
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context.\n"
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )

    train_max_length = 0

    for data in dataset:
        text = format_example(data)
        tokens = tokenizer.encode(text)
        if len(tokens) > train_max_length:
            train_max_length = len(tokens)

    print("Max Train Token Length: ", train_max_length)

    def tokenize(example, max_length):
        # 1. Full text
        text = format_example(example)

        tokens = tokenizer(
                        text,
                        truncation=True,
                        #max_length=512,
                        max_length=max_length + 1,
                        #padding="max_length",
                        padding=False
                 )
        
        pad_id = tokenizer.pad_token_id
        tokens["input_ids"].append(pad_id)
        tokens["attention_mask"].append(0)

        text_index = text.find("### Response:\n")
        prompt = text[:text_index]
        result = text[text_index:]
        #print("Prompt:\n", prompt)
        #print("Result:\n", result)

        prompt_tokens = tokenizer(prompt)
        result_tokens = tokenizer(result)
        #print("Tokenized Results: ", result_tokens)

        to_be_masked = len(prompt_tokens["input_ids"])
        #print("Amount of tokens to be masked: ", to_be_masked)

        labels = tokens["input_ids"].copy()
        
        #print(labels)
        #print(tokenizer.decode(labels))

        #get the index of the first padding
        first_pad = labels.index(pad_id) if pad_id in labels else None
        

        for i in range(len(labels)):
            #only mask
            if i < to_be_masked:
                #should be masked
                labels[i] = -100
            elif labels[i] == pad_id:
                #is a padding -> should also be masked
                labels[i] = -100

        #set the first padding to unmasked, to tell the model when to stop
        labels[first_pad] = pad_id


        tokens["labels"] = labels
        
        return tokens


    from transformers import AutoModelForCausalLM

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    raw_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=None,
        trust_remote_code=True,
        torch_dtype="auto",  # automatically use float16 if supported    
        #local_files_only=True
    )

    print("Loaded raw model..", flush=True)


    # Define LoRA configuration
    lora_config = LoraConfig(
        r=32,                 # rank of LoRA matrices
        lora_alpha=64,        # scaling factor
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj"], # this is experimental
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )


    tokenized_dataset = dataset.map(tokenize, batched=False, fn_kwargs={"max_length": train_max_length})


    #Attach fresh Lora Adapter last
    model = get_peft_model(raw_model, lora_config)
    print("Attached fresh Lora Adapter to base model")
    model.to(device)
    print("Moved Peft Model to: ", device)
    del raw_model


    from transformers import TrainingArguments
    from trl import SFTTrainer

    from math import ceil

    batch_size = 1
    acc_steps = args.acc_steps
    learning_rate = 1e-4


    # calculate total steps per epoch
    steps_per_epoch = ceil(len(tokenized_dataset) / (batch_size * acc_steps))

    # evaluate every half epoch
    eval_steps = ceil(steps_per_epoch *(1 / args.eval_steps_per_epoch))

    print("Eval and save steps: ", eval_steps)

    if args.eval_callback:
        eval_callback = DualEvalCallback(
            checkpoint_dir = args.checkpoint_dir_path,
            base_model_path = args.base_model_path,
            model_name = args.model_name,
            output_path = args.callback_output_path,

            data_path_1 = args.callback_data_path,
            eval_mode_1 = args.callback_eval_mode,
            comparison_path_1 = args.callback_comparison_path,

            data_path_2 = args.callback_data_path2,
            eval_mode_2 = args.callback_eval_mode2,
            comparison_path_2 = args.callback_comparison_path2,

            mongo_port1 = args.eval_port,
            mongo_port2 = args.eval_port2,
            eval_gpu = args.eval_gpu
        )


    # handle cpu vs gpu
    bf16 = False
    fp16 = False

    if device.type == "cuda":
        bf16 = torch.cuda.is_bf16_supported()
        fp16 = not bf16

    # Training arguments
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=acc_steps,
        learning_rate=learning_rate,
        bf16=bf16,
        fp16=fp16,
        num_train_epochs=args.epochs,
        logging_steps=ceil(eval_steps / 3),
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=100,
        warmup_ratio=0.03,
        report_to="none",
    )
    
    trainer = None

    if args.eval_callback:

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            callbacks=[eval_callback],
        )
    else:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )

    if True:
        trainer.train()

if __name__ == "__main__":
    main()
