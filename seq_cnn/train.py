import torch
import torch.nn as nn
import json
import numpy as np
from model import CNN_rand
import pdb

class TrainConfig:
    def __init__(self):
        self.batch_size = 100
        self.num_epochs = 50

def load_data(split):
    with open(f"{split}.json", "r") as file:
        data = json.load(file)
    return data

if __name__ == "__main__":

    torch.manual_seed(8151)

    # Dataset
    device = torch.device("cuda")
    train_data = load_data('train')
    eval_data = load_data('dev')
    test_data = load_data('test')
    print("> Dataset Loaded.")

    # Model
    model = CNN_rand().to(device)
    print("> Model Loaded.")

    # Optimizer
    optimizer = torch.optim.Adadelta(model.parameters())

    # Loss
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    print("> Training Started.")
    config = TrainConfig()
    prev_eval_loss = float('inf')
    for epoch in range(config.num_epochs):

        print(f"Epoch {epoch}:")

        # Train
        model.train()
        for idx, start in enumerate(range(0, len(train_data), config.batch_size)):

            batch_data = train_data[start: min(start + config.batch_size, len(train_data))]

            losses = []
            correct_cnt = 0

            optimizer.zero_grad()

            for text, label in batch_data:
                text = torch.tensor(text, dtype=torch.long)[None, :].to(device)
                label = torch.tensor(label).reshape(1).to(device)

                output = model(text)
                loss = loss_fn(output, label)
                losses.append(loss.item())
                max_idx = torch.max(output, dim=1).indices.item()
                if max_idx == label.item():
                    correct_cnt += 1
                loss.backward()
            
            optimizer.step()

            # print(f"Step: {idx} Loss: {np.mean(np.array(losses)):.6f} Accuracy: {float(correct_cnt)/float(len(batch_data)):.2f}")

        # Evaluation
        model.eval()
        losses = []
        correct_cnt = 0
        for idx in range(len(eval_data)):

            text, label = eval_data[idx]
            text = torch.tensor(text, dtype=torch.long)[None, :].to(device)
            label = torch.tensor(label).reshape(1).to(device)

            output = model(text)
            loss = loss_fn(output, label)

            losses.append(loss.item())
            max_idx = torch.max(output, dim=1).indices.item()
            if max_idx == label.item():
                correct_cnt += 1
        eval_avg_loss = np.mean(np.array(losses))
        print(f"Eval Epoch: {epoch} Loss: {eval_avg_loss:.6f} Accuracy: {float(correct_cnt)/float(len(eval_data)):.4f}")

        # Early stopping
        if eval_avg_loss > prev_eval_loss:
            print("> Early stopping.")
            break
        else:
            prev_eval_loss = eval_avg_loss

    
    # Testing
    model.eval()
    losses = []
    correct_cnt = 0
    for idx in range(len(test_data)):

        text, label = test_data[idx]
        text = torch.tensor(text, dtype=torch.long)[None, :].to(device)
        label = torch.tensor(label).reshape(1).to(device)

        output = model(text)
        loss = loss_fn(output, label)

        losses.append(loss.item())
        max_idx = torch.max(output, dim=1).indices.item()
        if max_idx == label.item():
            correct_cnt += 1

    print(f"Test Loss: {np.mean(np.array(losses)):.6f} Accuracy: {float(correct_cnt)/float(len(test_data)):.4f}")