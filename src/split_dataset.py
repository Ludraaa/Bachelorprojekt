import os
import argparse

###SPLITTING DATASET
from datasets import load_dataset

def main():
    
    parser = argparse.ArgumentParser(
            description="Split a dataset into dev and test subsets. The size can be controlled to enforce a certain ratio."
            )
    
    parser.add_argument(
                "--input_file_path",
                type=str,
                required=True,
                help="Path to the input dataset."
            )

    parser.add_argument(
            "--output_dir",
            type=str,
            required=True,
            help="Path to the directory where the split files should be written. This is created if it doesnt exist."
            )

    parser.add_argument(
            "--test_ratio",
            type=float,
            required=True,
            help="Controls the relative size of the dev split."
            )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Splitting data set..")

    # Target file paths
    dev_path = os.path.join(args.output_dir, "dev_split.jsonl")
    test_path = os.path.join(args.output_dir, "test_split.jsonl")

    print("Splitting dataset and saving new JSONL files...")
    dataset = load_dataset("json", data_files=os.path.join(args.input_file_path), split="train", cache_dir="/workspace/.hf-cache")
   
    test_valid = dataset.train_test_split(test_size=args.test_ratio, seed=42)

    dev_set = test_valid["train"]
    test_set = test_valid["test"]

    # Only split + save if they don’t already exist
    if not all(os.path.exists(p) for p in [dev_path, test_path]):
        # Save as JSONL
        dev_set.to_json(dev_path, lines=True)
        test_set.to_json(test_path, lines=True)
        
    print("Done")

if __name__ == "__main__":
    main()
