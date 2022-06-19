import pandas as pd

from typing import Optional
from dataclasses import dataclass, field

from transformers import TrainingArguments
from transformers.trainer_utils import IntervalStrategy

task_to_keys = {
	"cola": ("sentence", None),
	"mnli": ("premise", "hypothesis"),
	"mrpc": ("sentence1", "sentence2"),
	"qnli": ("question", "sentence"),
	"qqp": ("question1", "question2"),
	"rte": ("sentence1", "sentence2"),
	"sst2": ("sentence", None),
	"stsb": ("sentence1", "sentence2"),
	"wnli": ("sentence1", "sentence2"),
}

@dataclass
class CustomTrainingArguments(TrainingArguments):
	"""Customized TrainingArguments"""

	## Overwrite TrainingArguments default value
	output_dir: str = field(
		default="../result/IBM", 
		metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
	)
	evaluation_strategy: IntervalStrategy = field(
		default="epoch",
		metadata={"help": "The evaluation strategy to use."},
	)
	save_strategy: IntervalStrategy = field(
		default="epoch",
		metadata={"help": "The checkpoint save strategy to use."},
	)
	save_total_limit: Optional[int] = field(
		default=1,
		metadata={
			"help": (
				"Limit the total amount of checkpoints. "
				"Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints"
			)
		},
	)

	## Self-defined
	save_model_accord_to_metric: bool = field(
		default=False,
		metadata={"help": "Whether to save model according to metric (f1/acc) instead of loss."},
	)
	overall_results_path: Optional[str] = field(
		default=None,
		metadata={"help": "File path of overall results."}
	)
	best_checkpoint_path: Optional[str] = field(
		default=None, 
		metadata={"help": "Path of the directory that contains checkpoint you want to load."}
	)
	exp_name: Optional[str] = field(
		default=None, 
		metadata={"help": "Experiment name, also the name of output folder."}
	)
	weighted_loss: bool = field(
		default=False,
		metadata={"help": "Whether to train with weighted_loss or not."}
	)

@dataclass
class DataTrainingArguments:
	"""
	Arguments pertaining to what data we are going to input our model for training and eval.

	Using `HfArgumentParser` we can turn this class
	into argparse arguments to be able to specify them on
	the command line.
	"""

	task_name: Optional[str] = field(
		default=None,
		metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
	)
	#dataset_name: Optional[str] = field(
	#	default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
	#)
	dataset_config_name: Optional[str] = field(
		default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
	)
	max_seq_length: int = field(
		default=128,
		metadata={
			"help": "The maximum total input sequence length after tokenization. Sequences longer "
			"than this will be truncated, sequences shorter will be padded."
		},
	)
	overwrite_cache: bool = field(
		default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
	)
	pad_to_max_length: bool = field(
		default=True,
		metadata={
			"help": "Whether to pad all samples to `max_seq_length`. "
			"If False, will pad the samples dynamically when batching to the maximum length in the batch."
		},
	)
	max_train_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": "For debugging purposes or quicker training, truncate the number of training examples to this "
			"value if set."
		},
	)
	max_eval_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
			"value if set."
		},
	)
	max_predict_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
			"value if set."
		},
	)
	train_file: Optional[str] = field(
		default=None, metadata={"help": "A csv or a json file containing the training data."}
	)
	validation_file: Optional[str] = field(
		default=None, metadata={"help": "A csv or a json file containing the validation data."}
	)
	test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

	## Self-defined
	dataset_root: Optional[str] = field(
		default="../dataset/processed", 
		metadata={
			"help": "Root path of all your datasets."
		}
	)
	dataset_name: Optional[str] = field(
		default="IBM", 
		metadata={
			"help": "Our stance dataset name."
		}
	)
	model_name: Optional[str] = field(
		default=None, 
		metadata={
			"help": "Model name"
		}
	)
	num_labels: Optional[int] = field(
		default=2, 
		metadata={
			"help": "Number of labels in the dataset, should match real dataset."
		}
	)
	load_dataset_from_hub: bool = field(
		default=False,
		metadata={
			"help": "Whether to download dataset from hub, if true, use self.dataset_name"
		}
	)
	fold: int = field(
		default=None, 
		metadata={
			"help": "Which fold of dataset to use"
		}
	)
	target_wise: bool = field(
		default=False, 
		metadata={
			"help": "Whether to train on random 5-fold or target-wise dataset"
		}
	)

	def __post_init__(self):
		if self.task_name is not None:
			self.task_name = self.task_name.lower()
			if self.task_name not in task_to_keys.keys():
				raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
		elif self.dataset_name is not None:
			pass
		elif self.train_file is None or self.validation_file is None:
			raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
		else:
			train_extension = self.train_file.split(".")[-1]
			assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
			validation_extension = self.validation_file.split(".")[-1]
			assert (
				validation_extension == train_extension
			), "`validation_file` should have the same extension (csv or json) as `train_file`."

		## Adjust num_labels to different datasets
		self.num_labels = pd.read_csv("{}/{}/train.csv".format(self.dataset_root, self.dataset_name))["stance"].nunique()

		if self.fold is None:
			raise ValueError("Unknown data fold to train on, please specify the fold properly.")

@dataclass
class ModelArguments:
	"""
	Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
	"""

	model_name_or_path: str = field(
		metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
	)
	config_name: Optional[str] = field(
		default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
	)
	tokenizer_name: Optional[str] = field(
		default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
	)
	cache_dir: Optional[str] = field(
		default=None,
		metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
	)
	use_fast_tokenizer: bool = field(
		default=True,
		metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
	)
	model_revision: str = field(
		default="main",
		metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
	)
	use_auth_token: bool = field(
		default=False,
		metadata={
			"help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
			"with private models)."
		},
	)