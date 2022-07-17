import sys
import ast
import numpy as np


def parse_array(s):
    return np.array(ast.literal_eval(s))

def read_array():
    return parse_array(sys.stdin.readline())

def write_array(arr):
    print(repr(arr.tolist()))


def update_ft_weights(center_embeddings, context_embeddings, center_subwords, context_word, label, learning_rate):
    """
    center_embeddings - VocabSize x EmbSize
    context_embeddings - VocabSize x EmbSize
    center_subwords - list of ints - list of identifiers of n-grams contained in center word
    context_word - int - identifier of context word
    label - 1 if context_word is real, 0 if it is negative
    learning_rate - float > 0 - size of gradient step
    """
    center_embs = np.array([center_embeddings[subword].copy() for subword in center_subwords])
    center_emb = center_embs.mean(axis=0)
    context_emb = context_embeddings[context_word].copy()
    x = center_emb@context_emb
    logit = 1 / (1 + np.exp(-x))
    # loss = - label * np.log(logit) - (1 - label) * np.log(1 - logit)
    diff = (- label / logit + (1 - label) / (1 - logit)) * logit * (1 - logit)
    for subword in center_subwords:
        center_embeddings[subword] -= learning_rate * diff * context_emb / len(center_subwords)
    context_embeddings[context_word] -= learning_rate * diff * center_emb


center_embeddings = read_array()
context_embeddings = read_array()
center_subwords = read_array()
context_word = int(sys.stdin.readline().strip())
label = int(sys.stdin.readline().strip())
learning_rate = float(sys.stdin.readline().strip())
result_center_embeddings = read_array()
result_context_embeddings = read_array()

update_ft_weights(center_embeddings, context_embeddings,
                  center_subwords, context_word, label, learning_rate)

write_array(center_embeddings)
write_array(context_embeddings)

print(result_center_embeddings == center_embeddings)
print(result_context_embeddings == context_embeddings)