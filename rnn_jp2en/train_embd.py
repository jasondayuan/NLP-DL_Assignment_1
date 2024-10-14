import json
import torch
import torch.nn as nn
import numpy as np
from utils.model import CBOW
from utils.constants import JP_VOCAB_SIZE, EN_VOCAB_SIZE
import random
import pdb
from tqdm import tqdm

class TrainingConfig:
    def __init__(self):
        self.num_epochs = 30
        self.n_embd = 128
        self.vocab_size = {'jp': JP_VOCAB_SIZE, 'en': EN_VOCAB_SIZE}
        self.batch_size = 64
        self.lr = 2e-3
        self.context_size = 2
        self.max_length = 128

def transform_format(processed, lang, config=TrainingConfig()):

    data = []

    for jp_text, en_text in processed:

        if lang == 'jp':
            text = jp_text
        else:
            text = en_text
        
        for idx in range(config.context_size, len(text) - config.context_size):
            label = text[idx]
            context = []
            context.extend([text[i] for i in range(idx - config.context_size, idx)])
            context.extend([text[i] for i in range(idx + 1, idx + config.context_size + 1)])
            data.append([context, label])

    return data

def data_to_tensor(batch_data, device):
    contexts = []
    labels = []

    for context, label in batch_data:
        context = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
        contexts.append(context)
        labels.append(label)

    contexts = torch.cat(contexts, dim=0)
    labels = torch.tensor(labels, dtype=torch.long, device=device)

    return contexts, labels

if __name__ == "__main__":
    random.seed(8151)

    config = TrainingConfig()
    languages = ['jp', 'en']
    with open(f'./dataset/train_preprocessed.json', 'r') as file:
        train_processed = json.load(file)
    with open(f'./dataset/eval_preprocessed.json', 'r') as file:
        eval_processed = json.load(file)

    for lang in languages:

        print(f"> Language: {lang}")

        # Setup
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
        model = CBOW(config.n_embd, config.vocab_size[lang]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-5)
        loss_fn = nn.CrossEntropyLoss()

        # Process original sequences into format for CBOW training
        train_data = transform_format(train_processed, lang)
        eval_data = transform_format(eval_processed, lang)

        # Training loop
        prev_eval_loss = float('inf')
        for epoch in range(config.num_epochs):

            # print(f"> Epoch: {epoch}")
            random.shuffle(train_data)

            # Train
            model.train()
            for idx, start in tqdm(enumerate(range(0, len(train_data), config.batch_size))):

                # Data preparation
                batch_data = train_data[start: min(start + config.batch_size, len(train_data))]
                context, label = data_to_tensor(batch_data, device)

                # Training
                optimizer.zero_grad()
                output = model(context)
                loss = loss_fn(output, label)
                loss.backward()
                optimizer.step()

            # Eval
            model.eval()
            context, label = data_to_tensor(eval_data, device)
            output = model(context)
            loss = loss_fn(output, label)
            print(f"Epoch: {epoch} Loss: {loss.item():.6f}")
            if prev_eval_loss < loss.item():
                print("> Early stopping triggered.")
                break
            prev_eval_loss = loss.item()
        
            # Save trained embeddings
            torch.save(model.embdding.state_dict(), f'./embd/{lang}_embedding.pth')