import sys
import ast
import numpy as np


def read_list():
    return ast.literal_eval(sys.stdin.readline())

def parse_array(s):
    return np.array(ast.literal_eval(s))

def read_array():
    return parse_array(sys.stdin.readline())

def write_array(arr):
    print(repr(arr.tolist()))


def generate_ft_sgns_samples(text, window_size, vocab_size, ns_rate, token2subwords):
    """
    text - list of integer numbers - ids of tokens in text
    window_size - odd integer - width of window
    vocab_size - positive integer - number of tokens in vocabulary
    ns_rate - positive integer - number of negative tokens to sample per one positive sample
    token2subwords - list of lists of int - i-th sublist contains list of identifiers of n-grams for token #i (list of subword units)

    returns list of training samples (CenterSubwords, CtxWord, Label)
    """
    samples = []
    for i in range(len(text)):
        for j in range(i - window_size // 2, i + window_size // 2 + 1):
            if j < 0 or j >= len(text) or j == i:
                continue
            subwords = [text[i].tolist()] + token2subwords[text[i]]
            samples.append((subwords, text[j], 1))
            for _ in range(ns_rate):
                # negative samples
                samples.append((subwords, np.random.randint(0, vocab_size), 0))
    return samples


text = read_array()
window_size = int(sys.stdin.readline().strip())
vocab_size = int(sys.stdin.readline().strip())
ns_rate = int(sys.stdin.readline().strip())
token2subwords = read_list()

result = generate_ft_sgns_samples(text, window_size, vocab_size, ns_rate, token2subwords)

print(repr(result))

# arr = np.array([1,2,3])
# # np.array to list
# print(arr.tolist())
