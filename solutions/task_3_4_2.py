import sys
import ast
import numpy as np


def parse_array(s):
    return np.array(ast.literal_eval(s))

def read_array():
    return parse_array(sys.stdin.readline())

def write_array(arr):
    print(repr(arr.tolist()))


def update_w2v_weights(center_embeddings, context_embeddings, center_word, context_word, label, learning_rate):
    """
    center_embeddings - VocabSize x EmbSize
    context_embeddings - VocabSize x EmbSize
    center_word - int - identifier of center word
    context_word - int - identifier of context word
    label - 1 if context_word is real, 0 if it is negative
    learning_rate - float > 0 - size of gradient step
    """
    # your code here - update center_embeddings and context_embeddings inplace
    center_emb = center_embeddings[center_word].copy()
    context_emb = context_embeddings[context_word].copy()
    x = center_emb@context_emb
    logit = 1 / (1 + np.exp(-x))
    # loss = - label * np.log(logit) - (1 - label) * np.log(1 - logit)
    diff = (- label / logit + (1 - label) / (1 - logit)) * logit * (1 - logit)
    center_embeddings[center_word] -= learning_rate * diff * context_emb
    context_embeddings[context_word] -= learning_rate * diff * center_emb


center_embeddings = read_array()
context_embeddings = read_array()
center_word = int(sys.stdin.readline().strip())
context_word = int(sys.stdin.readline().strip())
label = int(sys.stdin.readline().strip())
learning_rate = float(sys.stdin.readline().strip())

update_w2v_weights(center_embeddings, context_embeddings,
                   center_word, context_word, label, learning_rate)

write_array(center_embeddings)
write_array(context_embeddings)