import ipdb
import pandas as pd
import argparse
from tqdm import tqdm

def parse_args():
	parser = argparse.ArgumentParser(description="Preprocess dataset.")

	parser.add_argument("--dataset", type=str, default="IBM")
	parser.add_argument("--raw_path", type=str, default="../dataset/raw")
	parser.add_argument("--processed_path", type=str, default="../dataset/processed")

	args = parser.parse_args()

	return args

def statistics(args):
	processed_file = "{}/{}/data.csv".format(args.processed_path, args.dataset)

	data_df = pd.read_csv(processed_file)

	print()
	print("# total rows: {}".format(len(data_df)))
	print("# target topics: {}".format(len(data_df["target"].unique())))
	
	print()
	print(data_df.groupby("stance", dropna=False).size())
	print()
	print(data_df.groupby("sentiment", dropna=False).size())
	print()
	print(data_df.groupby("split", dropna=False).size())
	print()

if __name__ == "__main__":
	args = parse_args()
	statistics(args)