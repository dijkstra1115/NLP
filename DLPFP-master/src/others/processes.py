import os
import ipdb
import logging
import numpy as np

## Get the same logger as in `main.py`
logger = logging.getLogger("__main__")

def train_process(data_args, training_args, last_checkpoint, train_dataset, trainer):
	checkpoint = None
	if training_args.resume_from_checkpoint is not None:
		checkpoint = training_args.resume_from_checkpoint
	elif last_checkpoint is not None:
		checkpoint = last_checkpoint
	train_result = trainer.train(resume_from_checkpoint=checkpoint)
	metrics = train_result.metrics
	max_train_samples = (
		data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
	)
	metrics["train_samples"] = min(max_train_samples, len(train_dataset))

	#trainer.save_model()  # Saves the tokenizer too for easy upload

	trainer.log_metrics("train", metrics)
	#trainer.save_metrics("train", metrics)
	#trainer.save_state()

	return trainer.best_checkpoint_path

def eval_process(data_args, training_args, raw_datasets, eval_dataset, trainer):
	logger.info("*** Evaluate ***")

	## Loop to handle MNLI double evaluation (matched, mis-matched)
	tasks = [data_args.task_name]
	eval_datasets = [eval_dataset]
	if data_args.task_name == "mnli":
		tasks.append("mnli-mm")
		eval_datasets.append(raw_datasets["validation_mismatched"])
		combined = {}

	for eval_dataset, task in zip(eval_datasets, tasks):
		metrics = trainer.evaluate(eval_dataset=eval_dataset)

		max_eval_samples = (
			data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
		)
		metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

		if task == "mnli-mm":
			metrics = {k + "_mm": v for k, v in metrics.items()}
		if task is not None and "mnli" in task:
			combined.update(metrics)

		trainer.log_metrics("eval", metrics)
		trainer.save_metrics("eval", combined if task is not None and "mnli" in task else metrics)

		## Write overall results (model, acc, f1-macro, ...)
		if not os.path.isfile(training_args.overall_results_path):
			with open(training_args.overall_results_path, "w") as fw:
				f1s = ["{:10s}".format("F1-{}".format(label_i)) for label_i in range(data_args.num_labels)]
				metrics2report = [
					"{:20s}".format("Model"), 
					"{:10s}".format("Fold-Type"),
					"{:4s}".format("Fold"),
					"{:10s}".format("Acc"),
					"{:10s}".format("F1-Macro")
				]
				metrics2report.extend(f1s)
				fw.write("{}\n".format("\t".join(metrics2report)))

		with open(training_args.overall_results_path, "a") as fw:
			f1s = ["{:<10.4f}".format(metrics["eval_f1_{}".format(label_i)]) for label_i in range(data_args.num_labels)]
			metrics2report = [
				"{:20s}".format(data_args.model_name[:20]),
				"{:10s}".format("Target" if data_args.target_wise else "5-Fold"),
				"{:4d}".format(data_args.fold), 
				"{:<10.4f}".format(metrics["eval_accuracy"]), 
				"{:<10.4f}".format(metrics["eval_f1_macro"])
			]
			metrics2report.extend(f1s)
			fw.write("{}\n".format("\t".join(metrics2report)))

def predict_process(is_regression, data_args, training_args, raw_datasets, predict_dataset, trainer):
	logger.info("*** Predict ***")

	# Loop to handle MNLI double evaluation (matched, mis-matched)
	tasks = [data_args.task_name]
	predict_datasets = [predict_dataset]
	if data_args.task_name == "mnli":
		tasks.append("mnli-mm")
		predict_datasets.append(raw_datasets["test_mismatched"])

	for predict_dataset, task in zip(predict_datasets, tasks):
		# Removing the `label` columns because it contains -1 and Trainer won't like that.
		predict_dataset = predict_dataset.remove_columns("label")
		predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
		predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

		output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{task}.txt")
		if trainer.is_world_process_zero():
			with open(output_predict_file, "w") as writer:
				logger.info(f"***** Predict results {task} *****")
				writer.write("index\tprediction\n")
				for index, item in enumerate(predictions):
					if is_regression:
						writer.write(f"{index}\t{item:3.3f}\n")
					else:
						item = label_list[item]
						writer.write(f"{index}\t{item}\n")
