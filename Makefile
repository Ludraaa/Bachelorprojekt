.PHONY: help help-adapt-dataset adapt-dataset help-split-dataset split-dataset help-eval eval help-train train help-kill-mongo kill-mongo

# -----------------------------------
# Default help target (first)
# -----------------------------------
help:
	@echo "Available targets:"
	@echo ""
	@echo " adapt-dataset          Convert a raw dataset into WikiSP format"
	@echo ""
	@echo " split-dataset	       Splits a dataset into two."
	@echo ""
	@echo " eval	       Evaluates a model's performance on a dataset."
	@echo ""
	@echo " train	       Trains a model on one or multiple datasets."
	@echo ""
	@echo " kill-mongo	       Kills the specified mongodb session."
	@echo "Use 'make help-<target>' for details on a specific target"

# -----------------------------------
# Detailed help for adapt-dataset
# -----------------------------------
help-adapt-dataset:
	@echo "adapt-dataset"
	@echo "  Description: Convert an input dataset from /extern/data to WikiSP jsonl/json format."
	@echo "    Run with '--adaptmode train' for training set conversion."
	@echo "    For dev/test splits, use '--adapt_mode test' after splitting. This expects training set conversion beforehand."
	@echo ""
	@echo "  (1) Files read:"
	@echo "      - [--input_file_path] (raw dataset input)"
	@echo ""
	@echo "  (2) Files produced:"
	@echo "      - /extern/output/[--jsonl_output_path] (converted JSONL)"
	@echo "      - /extern/output/[--json_output_path] (converted JSON)"
	@echo ""
	@echo "  (3) Approx time:"
	@echo "      - ~minutes to a few hours depending on dataset size"
	@echo ""
	@echo "  (4) Approx RAM & disk:"
	@echo "      - RAM: ~TODO!!!!!!!!!!!!!1GB depending on dataset size"
	@echo "      - Disk: usually <1GB of outputs unless huge dataset"
	@echo ""
	@echo "  (5) Args:"
	@echo "      --input_file_path: String - The path to the raw dataset input (jsonl)"
	@echo "      --jsonl_output_path: String - The file path where the jsonl converted output should be saved. This can once again be used as an input for this target, to convert it further."
	@echo "      --json_output_path: String - The file path where the json converted output should be saved."
	@echo "      --adapt_mode: String ['train', 'test'] - Whether to convert to training set or test set. Warning:: !!Conversion to test set expects a train set as the input!!"
	@echo ""


# -----------------------------------
# Dataset adaptation target
# -----------------------------------
input_file_path      ?= /extern/data/Datasets/Qald7/original_files/train.jsonl
jsonl_output_path	?= adapted_Qald7_train.jsonl
json_output_path   ?= adapted_Qald7_train.json
adapt_mode       ?= train

adapt-dataset:
	@echo "Running dataset adaptation:"
	@echo "  input_file_path:  $(input_file_path)"
	@echo "  jsonl_output_path:  /extern/output/$(jsonl_output_path)"
	@echo "  json_output_path:   /extern/output/$(json_output_path)"
	@echo "  adapt_mode:   $(adapt_mode)"
	/opt/venv/bin/python3.10 src/adapt_dataset.py \
	    --input_file_path $(input_file_path) \
	    --jsonl_output_path /extern/output/$(jsonl_output_path) \
	    --json_output_path /extern/output/$(json_output_path) \
	    --adapt_mode $(adapt_mode)


# -----------------------------------
# Detailed help for split-dataset
# -----------------------------------
help-split-dataset:
	@echo "split-dataset"
	@echo "  Description: Splits a dataset into two parts (can be train/test or dev/test, as long as its two)."
	@echo "    The size of the second split can be controlled using the --test_ratio flag (float from 0 to 1)."
	@echo ""
	@echo "  (1) Files read:"
	@echo "      - [--input_file_path] (raw dataset input)"
	@echo ""
	@echo "  (2) Files produced:"
	@echo "      - /extern/output/[--output_dir]/dev_split.jsonl (dev (or train) split)"
	@echo "      - /extern/output/[--output_dir]/test_split.jsonl (test split)"
	@echo ""
	@echo "  (3) Approx time:"
	@echo "      - Should be almost instant."
	@echo ""
	@echo "  (4) Approx RAM & disk:"
	@echo "      - RAM: ~TODO!!!!!!!!!!!!!1GB depending on dataset size"
	@echo "      - Disk: Exact same as input size, so usually low 100Mbs to few Gbs in extreme cases."
	@echo ""
	@echo "  (5) Args:"
	@echo "      --input_file_path: String - The path to the raw dataset input (jsonl)"
	@echo "      --output_dir: String - The file path to the directory, where the two output files will be saved to. See section '(2) Files produced' for further information."
	@echo "      --test_ratio: float [0..1] - The size ratio of the test split with respect to the dev/train split. Setting this to 0.8 will lead to: test: 0.8 and dev/train: 0.2 of the original input size."
	@echo ""

# -----------------------------------
# Dataset adaptation target
# -----------------------------------
# input_file_path reused from adapt_dataset
output_dir  ?= split_dataset/
test_ratio ?= 0.5

split-dataset:
	@echo "Running dataset split:"
	@echo "  input:  $(input_file_path)"
	@echo "  output directory:  /extern/output/$(output_dir)"
	@echo "  test ratio:   $(test_ratio)"
	/opt/venv/bin/python src/split_dataset.py \
	    --input_file_path $(input_file_path) \
	    --output_dir /extern/output/$(output_dir) \
	    --test_ratio $(test_ratio)

# -----------------------------------
# Detailed help for eval
# -----------------------------------
help-eval:
	@echo "eval"
	@echo "  Description: Evaluates given model on given Dataset. Calculates EM, F1, and can also compute EM with respect to another model's predictions or save the predictions.."
	@echo ""
	@echo "  (1) Files read:"
	@echo "      - [--checkpoint_dir] (path to the folder containing model files)"
	@echo "      - [--data_dir] (path to the folder containing dev and test set of the dataset (both have to exist. If you only have one of the two, duplicate and rename it appropriately! )"
	@echo "      - [--comparison_path] (path to a predictions file previously produced using this target (and the --save_path flag))"
	@echo ""
	@echo "  (2) Files produced:"
	@echo "      - /extern/output/[--save_path] if set. Otherwise, this target does not produce any files."
	@echo ""
	@echo "  (3) Approx time:"
	@echo "      - Varies greatly depending on Dataset size and model. This could take as little as 15 minutes or as much as 24 hours. Most times it takes about 30-60 minutes. The original WikiSP models take 10x as long as the retrained ones."
	@echo ""
	@echo "  (4) Approx RAM & disk:"
	@echo "      - RAM: ~TODO!!!!!!!!!!!!!1GB depending on dataset size"
	@echo "      - Disk: About as much as the input dataset, if saving is enabled. Otherwise none.."
	@echo ""
	@echo "  (5) Args:"
	@echo "      --checkpoint_dir: String - The path to the directory containing the model files."
	@echo "      --data_dir: String - The file path to the directory, where the datasets dev and test set are saved. See section '(1) Files read' for more information."
	@echo "      --eval_mode: String ['dev', 'test'] - whether to evaluate on the dev or test set of the dataset."
	@echo "      --get_current_results: bool (Default: True) - whether to evaluate against the dataset's saved gold results (possibly outdated), or the fetch gold results live from wikidata."
	@echo "      --save_path: String (Default: False) - whether to save the model's prediction (for later comparison)."
	@echo "      --comparison_path: String (Default: False) - whether to compare the model's predictions to a different model's predictions (previously saved using the --save_path flag of this target) in addition to the dataset's gold predictions."
	@echo "      --mongo_port: int - The port of the mongodb session to be used. Start (and kill)  this using the provided target."
	@echo ""



# -----------------------------------
# eval target
# -----------------------------------
checkpoint_dir      ?= model_dir
data_dir  ?= dataset_dir
eval_mode ?= dev
get_current_results ?= True
save_path ?= False
comparison_path ?= False
mongo_port ?= 27016

eval:
	@echo "Starting MongoDB for eval on port $(mongo_port). The relevant db file lives at /extern/output/Mongo/mongo_eval_db."
	mkdir -p /extern/output/Mongo/mongo_eval_db
	nohup /mongodb-linux-x86_64-ubuntu2204-7.0.5/bin/mongod --dbpath /extern/output/Mongo/mongo_eval_db --bind_ip localhost --port $(mongo_port) > /extern/output/Mongo/mongo_eval.log 2>&1 &
	@sleep 2
	@echo "Running eval:"
	@echo "  checkpoint_dir:  $(checkpoint_dir)"
	@echo "  data_dir: $(data_dir)"
	@echo "  eval_mode: $(eval_mode)"
	@echo "  get_current_results: $(get_current_results)"
	@echo "  save_path: $(save_path)"
	@echo "  comparison_path: $(comparison_path)"
	@echo "  mongo_port: $(mongo_port)"
	/opt/venv/bin/python src/eval.py \
	    --checkpoint_dir $(checkpoint_dir) \
	    --data_dir $(data_dir) \
	    --eval_mode $(eval_mode) \
	    --get_current_results $(get_current_results) \
	    --save_path $(save_path) \
	    --comparison_path $(comparison_path) \
	    --mongo_port $(mongo_port)
	@echo "The mongodb session will persist, so REFINED does not have to rerun. If you want to evaluate on a different dataset next, kill the mongo session using the 'kill-mongo' make target first."


# -----------------------------------
# Detailed help for train
# -----------------------------------
help-train:
	@echo "train"
	@echo "  Description: Train a model on the specified datasets. Has a builtin callback, that runs eval.py during training.."
	@echo ""
	@echo "  (1) Files read:"
	@echo "      - [--datasets] (training datasets)"
	@echo "      - [--base_model_path] (the files of the base model, that is to be used to train on)"
	@echo "      - optionally: [--callback_data_path] (dataset Nr. 1 to be evaluated against)"
	@echo "      - optionally: [--callback_comparison_path] (prediction file Nr. 1 to be evaluated against)"
	@echo "      - optionally: [--callback_data_path1] (dataset Nr. 2 to be evaluated against)"
	@echo "      - optionally: [--callback_comparison_path2] (prediction file Nr. 2 to be evaluated against)"
	@echo ""
	@echo "  (2) Files produced:"
	@echo "      - /extern/output/[--checkpoint_dir_path] (directory where the training checkpoints will be saved)"
	@echo "      - /extern/output/[--callback_output_path] (csv where the training results will be saved)"
	@echo ""
	@echo "  (3) Approx time:"
	@echo "      - ~a few hours to days, depending on parameters. Using a decently sized (~30k) dataset, 5 epochs and dual evaluation 4 times per epoch, it takes around one day."
	@echo ""
	@echo "  (4) Approx RAM & disk:"
	@echo "      - RAM: ~TODO!!!!!!!!!!!!!1GB depending on dataset size"
	@echo "      - Disk: Each combined model (from the checkpoint) takes up about 13GB."
	@echo "        Training for 5 epochs and saving/evaluating 4 times per epoch needs a total of 20 * 13GB" = 260GB for the models alone. The total storage needed seems to be around 300GB when accounting the checkpoints themselves as well.
	@echo "      - GPU: 1 x 46GB Vram for Training + 1 more for evaluation (optional, but recommended)"
	@echo ""
	@echo "  (5) Args:"
	@echo "      --checkpoint_dir_path: String - Path where the saved checkpoints and model files will go."
	@echo "      --datasets: String - Space seperated list of paths to train set files. For alpaca data, simply add 'Alpaca' as well. "
	@echo "      --scalings: String - Space seperated list of up/downsampling factors for each dataset. Length of datasets and scalings have to line up."
	@echo "      --model_name: String - Name used to save the models to be evaluated under. The corresponding checkpoint number will be appended to this. "
	@echo "      --base_model_path: String - Path to the directory containing the base model files. That model will be used for further training. "
	@echo "      --acc_steps: int - Gradient accumulation steps to be used for training. "
	@echo "      --epochs: int - Amount of epochs to be trained for. "
	@echo "      --eval_steps_per_epoch: int - Amount of evaluation steps during each epoch of training. If this is set to 4, there will be an eval callback (if enabled) and a checkpoint saved every 0.25 epochs. "
	@echo "      --eval_callback: bool - Whether to enable eval callback. This is highly encouraged. If for some reason training without eval callback is wanted, all saved checkpoints can later be evaluated just like using the eval callback by using the mass_eval.py script. "
	@echo "      --callback_output_path: String - Path to save the results of the eval callback to. This has to be a csv file. "
	@echo "      --callback_data_path: String - Path to the first dataset to be evaluated against. "
	@echo "      --callback_eval_mode: String ['dev', 'test'] - Whether to evaluate the first dataset's dev or test set. "
	@echo "      --callback_comparison_path: String - Path to the first predicted results fine to be used for comparison. This is optional. "
	@echo "      --callback_data_path2: String - Path to the second dataset to be evaluated against. This is optional. "
	@echo "      --callback_eval_mode2: String ['dev', 'test'] - Whether to evaluate the second dataset's dev or test set. This has to be set if '--callback_data_path2' is set. "
	@echo "      --callback_comparison_path2: String - Path to the first predicted results fine to be used for comparison. This is optional. "
	@echo "      --eval_port: int - Port for the first eval session's mongodb to run on. "
	@echo "      --eval_port_2: int - Port for the second eval session's mongodb to run on. "
	@echo ""


# -----------------------------------
# train target
# -----------------------------------
checkpoint_dir_path ?= /Models/MyWikiSP_WWQ_Q7_LittleAlpaca
datasets ?= /extern/data/Datasets/WikiWebQuestions/TrainingData/train.json /extern/data/Datasets/Qald7/TrainingData/train.json Alpaca
scalings ?= 5 20 0.1
model_name ?= MyWikiSP_WWQ_Q7_LittleAlpaca-
base_model_path ?= /extern/data/Models/Llama-2-7b-hf
acc_steps ?= 16
epochs ?= 5
eval_steps_per_epoch ?= 2
eval_callback ?= True
callback_output_path ?= /TrainingResults/MyWikiSP_WWQ_Q7_LittleAlpaca.csv
callback_data_path ?= /extern/data/Datasets/WikiWebQuestions
callback_eval_mode ?= dev
callback_comparison_path ?= /extern/data/PredictedResults/local_wikisp_q7_wwq_dev.jsonl
callback_data_path2 ?=
callback_eval_mode2 ?=
callback_comparison_path2 ?=
eval_port ?= 27017
eval_port2 ?= 27018


train:
	@echo "Starting MongoDB for eval callback on port $(eval_port). The relevant db file lives at /extern/output/Mongo/eval_callback_1_db."
	mkdir -p /extern/output/Mongo/eval_callback_1_db  && 
	nohup /mongodb-linux-x86_64-ubuntu2204-7.0.5/bin/mongod --dbpath /extern/output/Mongo/eval_callback_1_db --bind_ip localhost --port $(eval_port) > /extern/output/Mongo/eval_callback_1.log 2>&1 &
	
	@echo "Starting MongoDB for eval callback on port $(eval_port2). The relevant db file lives at /extern/output/Mongo/eval_callback_2_db."
	
	mkdir -p /extern/output/Mongo/eval_callback_2_db &&
	nohup /mongodb-linux-x86_64-ubuntu2204-7.0.5/bin/mongod --dbpath /extern/output/Mongo/eval_callback_2_db --bind_ip localhost --port $(eval_port2) > /extern/output/Mongo/eval_callback_2.log 2>&1 &
	
	@sleep 2
	@echo "Running train.py:"
	@echo "  checkpoint_dir_path: /extern/output/$(checkpoint_dir_path)"
	@echo "  datasets: $(datasets)"
	@echo "  scalings: $(scalings)"
	@echo "  model_name: $(model_name)"
	@echo "  base_model_path: $(base_model_path) "
	@echo "  acc_steps: $(acc_steps) "
	@echo "  epochs: $(epochs) "
	@echo "  eval_steps_per_epoch: $(eval_steps_per_epoch) "
	@echo "  eval_callback: $(eval_callback) "
	@echo "  callback_output_path: /extern/output/$(callback_output_path) "
	@echo "  callback_data_path: $(callback_data_path) "
	@echo "  callback_eval_mode: $(callback_eval_mode) "
	@echo "  callback_comparison_path: $(callback_comparison_path) "
	@echo "  callback_data_path2: $(callback_data_path2) "
	@echo "  callback_eval_mode2: $(callback_eval_mode2) "
	@echo "  callback_comparison_path2: $(callback_comparison_path2) "
	@echo "  eval_port: $(eval_port) "
	@echo "  eval_port2: $(eval_port2) "
	/opt/venv/bin/python src/train.py \
	    --checkpoint_dir_path /extern/output/$(checkpoint_dir_path) \
	    --datasets $(datasets) \
	    --scalings $(scalings) \
	    --model_name $(model_name) \
	    --base_model_path $(base_model_path) \
	    --acc_steps $(acc_steps) \
	    --epochs $(epochs) \
	    --eval_steps_per_epoch $(eval_steps_per_epoch) \
	    --eval_callback $(eval_callback) \
	    --callback_output_path /extern/output/$(callback_output_path) \
	    --callback_data_path $(callback_data_path) \
	    --callback_eval_mode $(callback_eval_mode) \
	    --callback_comparison_path $(callback_comparison_path) \
	    --callback_data_path2 $(callback_data_path2) \
	    --callback_eval_mode2 $(callback_eval_mode2) \
	    --callback_comparison_path2 $(callback_comparison_path2) \
	    --eval_port $(eval_port) \
	    --eval_port2 $(eval_port2)
	
	@echo "Cleaning up DBs.."
	rm -rf /extern/output/Mongo/eval_callback_1_db
	rm -rf /extern/output/Mongo/eval_callback_2_db
	@echo "All Done! Check the training results in $(callback_output_path) if callback was enabled. The corresponding checkpoints can be found in $(checkpoint_dir_path)."

# -----------------------------------
# Detailed help for kill-mongo
# -----------------------------------
help-kill-mongo:
	@echo "kill-mongo"
	@echo "  Description: Kills a specific mongodb session."
	@echo ""
	@echo "  (1) Files read:"
	@echo "      - [--db_dir_path] (db file)"
	@echo ""
	@echo "  (2) Files produced:"
	@echo "      - None"
	@echo ""
	@echo "  (3) Approx time:"
	@echo "      - Should be almost instant."
	@echo ""
	@echo "  (4) Approx RAM & disk:"
	@echo "      - RAM: Insignificant."
	@echo "      - Disk: None."
	@echo ""
	@echo "  (5) Args:"
	@echo "      --db_dir_path: String - The path to the db directory"
	@echo ""

# -----------------------------------
# kill-mongo target
# -----------------------------------

db_dir_path ?= /extern/output/Mongo/mongo_eval_db

kill-mongo:
	@echo "Running kill-mongo:"
	@echo "  db_dir_path:  $(db_dir_path)"
	@rm -rf $(db_dir_path)
	@echo "Done!"

