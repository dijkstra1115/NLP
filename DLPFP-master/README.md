# DLP 2022 Final Project
## Dependencies
```
ipdb
tqdm
pandas
scipy
sklearn

*** pytorch ***
torch==1.6.0+cu101

*** huggingface packages ***
datasets
transformers==4.19.0.dev0
```
Install pytorch as follows. (not necessary same version as mine)
```
$ pip install torch==1.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```
Install huggingface transformers **from source** as follows.
```
$ pip install git+https://github.com/huggingface/transformers
```

## Dataset
Create the following directory `dataset/processed/{$YOUR_DATASET_NAME}` at the same level as `src`. For random 5-fold, put your `train.csv` and `test.csv` into `dataset/processed/{$YOUR_DATASET_NAME}/split_{$FOLD_INDEX}`. For target-wise, put your `train.csv` and `test.csv` into `dataset/processed/{$YOUR_DATASET_NAME}/target_{$FOLD_INDEX}`.

## Usage
Run the following command to train on your dataset.
```
$ python main.py \
	--model_name_or_path bert-base-uncased \
	--dataset_name $YOUR_DATASET_NAME \
	--train_file train.csv \
	--validation_file test.csv \
	--do_train \
	--do_eval \
	--max_seq_length 128 \
	--per_device_train_batch_size 32 \
	--learning_rate 2e-5 \
	--num_train_epochs 10 \
	--output_dir ../result/$YOUR_DATASET_NAME \
	--overwrite_output_dir \
	--save_model_accord_to_metric
```
Note that you could also tweak the arguments to your needs.

You can also write your command in the file `script.sh` and execute the following line.
```
$ sh script.sh
```