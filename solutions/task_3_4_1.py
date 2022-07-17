import sys
import ast
import numpy as np


def parse_array(s):
    return np.array(ast.literal_eval(s))

def read_array():
    return parse_array(sys.stdin.readline())

def write_array(arr):
    print(repr(arr.tolist()))


def generate_w2v_sgns_samples(text, window_size, vocab_size, ns_rate):
    """
    text - list of integer numbers - ids of tokens in text
    window_size - odd integer - width of window
    vocab_size - positive integer - number of tokens in vocabulary
    ns_rate - positive integer - number of negative tokens to sample per one positive sample

    returns list of training samples (CenterWord, CtxWord, Label)
    """
    samples = []
    for i in range(len(text)):
        for j in range(i - window_size // 2, i + window_size // 2 + 1):
            if j < 0 or j >= len(text) or j == i:
                continue
            samples.append((text[i], text[j], 1))
            for _ in range(ns_rate):
                # negative samples
                samples.append((text[i], np.random.randint(0, vocab_size), 0))
    return samples


# text = read_array()
# window_size = int(sys.stdin.readline().strip())
# vocab_size = int(sys.stdin.readline().strip())
# ns_rate = int(sys.stdin.readline().strip())
text = [1,2,3,4]
window_size = 3
vocab_size = 5
ns_rate = 2

result = generate_w2v_sgns_samples(text, window_size, vocab_size, ns_rate)

write_array(np.array(result))