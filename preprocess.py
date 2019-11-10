import os
import pandas as pd
import re
import sys
from tqdm import tqdm

BLOG = 'blog'
CLASS = 'class'

def load_data(file_in, folder='./data/', has_labels=False):
	path = folder + file_in
	dataset = pd.read_csv(path, names=[BLOG, CLASS])
	if has_labels:
		return dataset[BLOG].to_numpy(), dataset[CLASS].to_numpy()
	else:
		return dataset[BLOG].to_numpy()

def store_as_text(data, file_out, folder='./data/'):
	path = folder + file_out
	with open(path, 'w+') as file:
		print(len(data))
		for d in tqdm(data):
			file.write(d + '\n')

def store_as_csv(file_in, file_out, labels=None, folder='./out/'):
	path = folder + file_in
	with open(path, 'r') as file:
		preprocessed_data = file.readlines()

	print(len(preprocessed_data))

	if labels is not None:
		df = pd.DataFrame({BLOG: preprocessed_data, CLASS: labels})
	else:
		df = pd.DataFrame({BLOG: preprocessed_data})

	path = folder + file_out
	df.to_csv(path, header=False, index=False)


if __name__ == "__main__":

	if len(sys.argv) > 1:
		n_lines = int(sys.argv[1])
	else:
		n_lines = 10000
	
	# train data
	data, labels = load_data('train_posts.csv', has_labels=True)
	store_as_text(data, 'data_train.txt')
	os.system(f'bash "script-preprocess" train {n_lines}')
	store_as_csv('data_train.out', 'data_train_preprocessed.csv', labels=labels[:n_lines])
	
	# test data
	data = load_data('test_split01.csv')
	store_as_text(data, 'data_test.txt')
	os.system(f'bash "script-preprocess" test {n_lines}')
	store_as_csv('data_test.out', 'data_test_preprocessed.csv')
