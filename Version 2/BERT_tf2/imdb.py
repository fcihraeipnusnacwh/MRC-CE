import os, re
import tensorflow as tf
import pandas as pd
import numpy as np

def load_directory_data(directory):
	data = {}
	data["sentence"] = []
	data["sentiment"] = []
	for file_path in os.listdir(directory):
		with tf.io.gfile.GFile(os.path.join(directory, file_path), "r") as f:
			data["sentence"].append(f.read())
			data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
	return pd.DataFrame.from_dict(data)

# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory):
	pos_df = load_directory_data(os.path.join(directory, "pos"))
	neg_df = load_directory_data(os.path.join(directory, "neg"))
	pos_df["polarity"] = 1
	neg_df["polarity"] = 0
	return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)

# Download and process the dataset files.
def download_and_load_datasets(force_download=False):
	dataset = tf.keras.utils.get_file(fname="aclImdb.tar.gz", 
				origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", extract=True)
	train_df = load_dataset(os.path.join(os.path.dirname(dataset), "aclImdb", "train"))
	test_df = load_dataset(os.path.join(os.path.dirname(dataset), "aclImdb", "test"))
	return train_df, test_df

train_df, test_df = download_and_load_datasets()
print(train_df.head())

def convert_1str(df, max_seq_len):
	text = df['sentence'].tolist()
	text = [' '.join(t.split()[:max_seq_len]) for t in text]
	label = df['polarity'].tolist()
	return text, label

def GetImdbData(max_seq_len=256, max_num=99999):
	(train_text, train_label), (test_text, test_label) = map(lambda x:convert_1str(x, max_seq_len), (train_df[:max_num], test_df[:max_num]))
	(train_label, test_label) = map(lambda x:np.asarray(x, dtype='int32'), (train_label, test_label))
	return (train_text, train_label), (test_text, test_label)
