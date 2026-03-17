#CELL 1
#Machine Learning Code summarization Project by Marcos Fernandez and Juan Carrion
!pip install torch==2.2.2 torchtext==0.17.2 portalocker==2.7.0 datasets pandas
import torch
import re
from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

#We load the dataset
raw_dataset = load_dataset("flytech/python-codes-25k", split='train[:20000]')
# We split data into 3: 80% Train, 10% Val, 10% Test
train_test_split = raw_dataset.train_test_split(test_size=0.2, seed=42)
test_valid_split = train_test_split['test'].train_test_split(test_size=0.5, seed=42)

train_data_raw = train_test_split['train']
val_data_raw = test_valid_split['train']
test_data_raw = test_valid_split['test']

def process_pairs(dataset):
    pairs = []
    for ex in dataset:
        if ex['output'] and ex['instruction']:#in this dataset,output contains de code and instruction the summary of the code
            pairs.append((ex['output'], ex['instruction']))
    return pairs

train_data = process_pairs(train_data_raw)
val_data = process_pairs(val_data_raw)
test_data = process_pairs(test_data_raw)

# Tokenizers
tokenizer_summary = get_tokenizer('basic_english')
def tokenizer_code(text):

    tokens = re.findall(r"[\w']+|[^\w\s]", text)
    return tokens

def yield_tokens(data_iter, is_code):
    for code, summary in data_iter:
        yield tokenizer_code(code) if is_code else tokenizer_summary(summary)

#Vocab
special_symbols = ['<unk>', '<pad>', '<sos>', '<eos>']

vocab_code = build_vocab_from_iterator(yield_tokens(train_data, True), specials=special_symbols, min_freq=2)
vocab_summary = build_vocab_from_iterator(yield_tokens(train_data, False), specials=special_symbols, min_freq=2)

vocab_code.set_default_index(vocab_code["<unk>"])
vocab_summary.set_default_index(vocab_summary["<unk>"])

print(f"Dataset ready: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")