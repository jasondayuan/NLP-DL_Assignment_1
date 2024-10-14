import json
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils.constants import JP_VOCAB_SIZE, EN_VOCAB_SIZE
import random
import pdb

class EvalConfig:
    def __init__(self):
        self.n_embd = 128
        self.vocab_size = {'jp': JP_VOCAB_SIZE, 'en': EN_VOCAB_SIZE}

def most_similar(lang, token, k=5):
    config = EvalConfig()

    embedding = nn.Embedding(config.vocab_size[lang], config.n_embd)
    embedding.load_state_dict(torch.load(f"./embd/{lang}_embedding.pth"))
    embeddings = torch.arange(0, config.vocab_size[lang], dtype=torch.long)
    embeddings = embedding(embeddings) 

    with open(f'./dataset/vocab_{lang}.json', 'r', encoding='utf-8') as file:
        vocab_dict = json.load(file)
    idx_to_tok = {value: key for key, value in vocab_dict.items()}
    
    assert token in vocab_dict
    token_num = vocab_dict[token]

    cosine_sim = torch.zeros(config.vocab_size[lang])
    for idx in range(config.vocab_size[lang]):
        if idx == token_num:
            continue
        cos = F.cosine_similarity(embeddings[token_num], embeddings[idx], dim=0)
        cosine_sim[idx] = cos
    
    topk_values, topk_indices = torch.topk(cosine_sim, k)
    topk_tokens = []
    for idx in topk_indices:
        topk_tokens.append(idx_to_tok[idx.item()])

    return (topk_values, topk_tokens)

def analogy(lang, pos, neg):
    config = EvalConfig()

    embedding = nn.Embedding(config.vocab_size[lang], config.n_embd)
    embedding.load_state_dict(torch.load(f"./embd/{lang}_embedding.pth"))
    embeddings = torch.arange(0, config.vocab_size[lang], dtype=torch.long)
    embeddings = embedding(embeddings) 

    with open(f'./dataset/vocab_{lang}.json', 'r', encoding='utf-8') as file:
        vocab_dict = json.load(file)

    pos[0] = embeddings[vocab_dict[pos[0]]]
    pos[1] = embeddings[vocab_dict[pos[1]]]
    neg[0] = embeddings[vocab_dict[neg[0]]]
    neg[1] = embeddings[vocab_dict[neg[1]]]

    return F.cosine_similarity(pos[0] - pos[1], neg[0] - neg[1], dim=0)
    
    
if __name__ == "__main__":
    res = most_similar('jp', '僕')
    print(res)