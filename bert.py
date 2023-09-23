# from transformers import AutoTokenizer, AutoModel
# from torch import nn

# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = AutoModel.from_pretrained("bert-base-uncased")

# inputs = tokenizer("Hello world!", return_tensors="pt")
# outputs = model(**inputs)
# print(outputs)
from datasets import load_dataset
data_file = ""
dataset = load_dataset('oscar', 'unshuffled_deduplicated_it',cache_dir='~/hardDisk/DeepLearning/bert',)
