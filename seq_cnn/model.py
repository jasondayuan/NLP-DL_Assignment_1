import torch
import torch.nn as nn
import pdb

class CNNConfig:
    def __init__(self):
        self.n_embd = 256
        self.vocab_size = 11705
        self.n_class = 4
        self.min_seq_len = 10

class CNN_rand(nn.Module):
    def __init__(self, config=CNNConfig()):
        super(CNN_rand, self).__init__()

        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.conv_layer = nn.Conv2d(in_channels=1, out_channels=32, 
                                    kernel_size=(3, config.n_embd), padding=(config.min_seq_len // 2, 0))
        self.linear = nn.Linear(32, config.n_class)

    def forward(self, x):
        x = self.embedding(x)[:, None, :, :]
        x = self.conv_layer(x)
        B, C, H, _ = x.shape
        x = x.reshape(B, C, H)
        x, _ = torch.max(x, dim=-1)
        x = self.linear(x)
        return x