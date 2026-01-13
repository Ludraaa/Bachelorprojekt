#!/usr/bin/env python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
import os
import re

def main():
    parser = argparse.ArgumentParser(description="Merge a LoRA checkpoint into a base model.")

    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="Path to the base model (e.g., Llama-2-7b)."
    )
    parser.add_argument(
        "--lora_checkpoint_path",
        type=str,
        required=True,
        help="Path to the LoRA checkpoint folder."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model. This will be attached before the checkpoint number."
    )

    args = parser.parse_args()

    base_model_path = args.base_model_path
    lora_checkpoint_path = args.lora_checkpoint_path
    model_name = args.model_name

    print(f"Loading base model from {base_model_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print(f"Loading LoRA adapter from {lora_checkpoint_path}...")
    model_with_lora = PeftModel.from_pretrained(base_model, lora_checkpoint_path)

    print("Merging LoRA adapter into the base model...")
    merged_model = model_with_lora.merge_and_unload()
    
    # lora_checkpoint_path = "./models/MyModels/MyWikiSP/my_wikisp_alpaca/checkpoint-1755"
    # model_name = "MyWikiSP"

    #parent of checkpoint folder (one level up)
    checkpoint_parent = os.path.dirname(lora_checkpoint_path)  # .../my_wikisp_alpaca

    #base name of checkpoint folder
    checkpoint_basename = os.path.basename(lora_checkpoint_path)  # "checkpoint-1755"

    #extract digits at the end
    match = re.search(r'\d+$', checkpoint_basename)
    if match:
        checkpoint_number = match.group(0)
    else:
        raise ValueError(f"No digits found in checkpoint folder name: {checkpoint_basename}")

    #construct merged output path
    merged_output_path = os.path.join(checkpoint_parent, f"{model_name}{checkpoint_number}")

    #Make sure folder exists
    os.makedirs(merged_output_path, exist_ok=True)



    print(f"Saving merged model to {merged_output_path}...")
    merged_model.save_pretrained(merged_output_path)

    print(f"Saving tokenizer from base model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(merged_output_path)

    print("Done!")

if __name__ == "__main__":
    main()
