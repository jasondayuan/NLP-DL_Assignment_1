import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from utils.constants import JP_VOCAB_SIZE, EN_VOCAB_SIZE, SOS_TOKEN
import math
import pdb

class CBOW(nn.Module):
    def __init__(self, n_embd, vocab_size):
        super(CBOW, self).__init__()
        self.embdding = nn.Embedding(vocab_size, n_embd)
        self.linear = nn.Linear(n_embd, vocab_size)
    
    def forward(self, x): # (B, context_size * 2)
        x = self.embdding(x) # (B, context_size * 2, n_embd)
        x = torch.mean(x, dim=1) # (B, n_embd)
        x = self.linear(x) # (B, vocab_size)
        return x

class RNNConfig:
    def __init__(self):
        self.n_embd = 128
        self.hidden_size = 256
        self.max_len = 72

class EncoderRNN(nn.Module):
    def __init__(self, config=RNNConfig()):
        super(EncoderRNN, self).__init__()
        self.embedding = nn.Embedding(JP_VOCAB_SIZE, config.n_embd)
        self.lstm = nn.LSTM(config.n_embd, config.hidden_size)

    def forward(self, x, seq_lens):
        x = self.embedding(x)
        x = pack_padded_sequence(x, seq_lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (h_n, c_n) = self.lstm(x)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        return output, (h_n, c_n)

class DecoderAttnRNN(nn.Module):
    def __init__(self, config=RNNConfig()):
        super(DecoderAttnRNN, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(EN_VOCAB_SIZE, config.n_embd)
        self.lstm = nn.LSTM(config.n_embd, config.hidden_size, batch_first=True)
        self.linear = nn.Linear(config.hidden_size * 2, EN_VOCAB_SIZE)

    def forward(self, target_tensor, encoder_outputs, encoder_seq_lens, h0, c0):
        '''
        target_tensor - (B, max_len)
        encoder_outputs - (B, #max_encoder_seq_len, hidden_size)
        encoder_seq_lens - (B, )
        c0, h0 - (1, B, hidden_size)
        '''
        attn_mask = self.get_attn_mask(encoder_seq_lens)
        B = encoder_seq_lens.shape[0]
        device = encoder_seq_lens.device
        decoder_input = torch.empty((B, 1), dtype=torch.long, device=device).fill_(SOS_TOKEN) # (B, 1)
        decoder_hidden = h0
        decoder_cell = c0
        decoder_outputs = []

        for idx in range(self.config.max_len):
            if target_tensor is not None: # teacher forcing
                decoder_input = target_tensor[:, idx].unsqueeze(1) # (B, 1)
                decoder_output, (decoder_hidden, decoder_cell) = self.forward_step(encoder_outputs, decoder_input, decoder_hidden, decoder_cell, attn_mask)
                    # decoder_output (B, 1, EN_VOCAB_SIZE)
                decoder_outputs.append(decoder_output)
            else : # autoregressive generation
                decoder_output, (decoder_hidden, decoder_cell) = self.forward_step(encoder_outputs, decoder_input, decoder_hidden, decoder_cell, attn_mask)
                _, topi = decoder_output.topk(1, dim=-1) # topi (B, 1, 1)
                decoder_input = topi.squeeze(-1).detach()
                decoder_outputs.append(decoder_input)

        decoder_outputs = torch.cat(decoder_outputs, dim=1)

        return decoder_outputs

    
    def forward_step(self, encoder_outputs, decoder_input, decoder_hidden, decoder_cell, attn_mask):
        decoder_input = self.embedding(decoder_input) # (B, 1, n_embd)

        lstm_output, (lstm_hidden, lstm_cell) = self.lstm(decoder_input, (decoder_hidden, decoder_cell)) # lstm_output - (B, 1, hidden_size)
        
        # Attention
        # query - lstm_hidden key - encoder_outputs value - encoder_outputs
        attn_score = (lstm_output @ encoder_outputs.transpose(-2, -1)) * (1.0 / math.sqrt(self.config.hidden_size)) # (B, 1, #max_encoder_seq_len)
        attn_score.masked_fill_(attn_mask == 0, float('-inf'))
        attn_dist = F.softmax(attn_score, dim=-1) # (B, 1, #max_encoder_seq_len)
        attn_output = attn_dist @ encoder_outputs # (B, 1, hidden_size)
        
        linear_input = torch.cat((lstm_output, attn_output), dim=-1) # (B, 1, hidden_size * 2)
        decoder_output = self.linear(linear_input) # (B, 1, EN_VOCAB_SIZE)

        return decoder_output, (lstm_hidden, lstm_cell)

    def get_attn_mask(self, encoder_seq_lens):
        B = encoder_seq_lens.shape[0]
        max_encoder_seq_len = torch.max(encoder_seq_lens)
        device = encoder_seq_lens.device
        mask = torch.zeros((B, 1, max_encoder_seq_len), device=device)
        for idx, length in enumerate(encoder_seq_lens):
            mask[idx, 0, :length] = 1
        return mask