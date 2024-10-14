import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from utils.model import EncoderRNN, DecoderAttnRNN, RNNConfig
from utils.constants import EOS_TOKEN
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pdb

class TrainingConfig():
    def __init__(self):
        self.num_epochs = 30
        self.batch_size = 64
        self.max_len = 72
        self.eval_sample_num = 1

def nll_calc(next_token_preds, next_tokens):
    '''
    next_token_preds - logits (#len, EN_VOCAB_SIZE)
    next_tokens (#len, )
    '''
    next_token_probs = F.softmax(next_token_preds, dim=-1)
    correct_probs = next_token_probs[range(next_token_probs.shape[0]), next_tokens]
    nll = -torch.sum(torch.log(correct_probs))
    return nll

def data_proc(split, device, config=RNNConfig()):
    '''
    Pads the sequences and transform them to tensors
    '''
    with open(f'./dataset/{split}_preprocessed.json', 'r') as file:
        data = json.load(file)
    
    # Assumes PAD_TOKEN == 0
    jp_data = torch.zeros((len(data), config.max_len + 1), dtype=torch.long, device=device)
    en_data = torch.zeros((len(data), config.max_len + 1), dtype=torch.long, device=device)

    print(f"> {split} data processing...")
    for idx, (jp_seq, en_seq) in tqdm(enumerate(data)):
        jp_data[idx, :len(jp_seq)] = torch.tensor(jp_seq, dtype=torch.long, device=device)
        jp_data[idx, -1] = torch.tensor(len(jp_seq), dtype=torch.long, device=device)
        en_data[idx, :len(en_seq)] = torch.tensor(en_seq, dtype=torch.long, device=device)
        en_data[idx, -1] = torch.tensor(len(en_seq), dtype=torch.long, device=device)
        # the last element of each row is the length of the original sequence
    
    return torch.cat((jp_data, en_data), dim=1) # (B, max_len * 2 + 2)

if __name__ == '__main__':

    # Vocab
    with open("./dataset/vocab_jp.json", 'r', encoding='utf-8') as file:
        jp_vocab_dict = json.load(file)
    with open("./dataset/vocab_en.json", 'r', encoding='utf-8') as file:
        en_vocab_dict = json.load(file)
    jp_idx_to_token = {value:key for key, value in jp_vocab_dict.items()}
    en_idx_to_token = {value:key for key, value in en_vocab_dict.items()}

    # Dataset
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    train_data = data_proc('train', device)
    eval_data = data_proc('eval', device)
    test_data = data_proc('test', device)

    # Model
    encoder = EncoderRNN().to(device)
    decoder = DecoderAttnRNN().to(device)
    encoder_optimizer = torch.optim.Adagrad(encoder.parameters(), lr=0.02)
    decoder_optimizer = torch.optim.Adagrad(decoder.parameters(), lr=0.02)
    encoder.embedding.load_state_dict(torch.load("./embd/jp_embedding.pth"))
    decoder.embedding.load_state_dict(torch.load("./embd/en_embedding.pth"))
    loss_fn = nn.CrossEntropyLoss()
    
    # Training loop
    config = TrainingConfig()
    prev_eval_loss = float('inf')
    for epoch in range(config.num_epochs):

        # Shuffle dataset
        train_data = train_data[torch.randperm(train_data.size(0))]
        
        # Train
        encoder.train()
        decoder.train()
        for start in tqdm(range(0, len(train_data), config.batch_size)):

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
        
            # Encoder 
            # encoder_outputs (B, #longest_len, hidden_size)
            # encoder_hn (1, B, hidden_size)
            # encoder_cn (1, B, hidden_size)
            batch_input = train_data[start:min(start+config.batch_size, len(train_data)), :] # (B, max_len * 2 + 2)
            encoder_input = batch_input[:, :config.max_len] # (B, max_len)
            encoder_seq_lens = batch_input[:, config.max_len] # (B, )
            encoder_outputs, (encoder_hn, encoder_cn) = encoder(encoder_input, encoder_seq_lens)

            # Decoder
            target_tensor = batch_input[:, config.max_len + 1: -1] # (B, max_len)
            target_lens = batch_input[:, -1] # (B, )
            decoder_outputs = decoder(target_tensor, encoder_outputs, encoder_seq_lens, encoder_hn, encoder_cn) # (B, max_len, EN_VOCAB_SIZE)

            # Loss
            next_token_preds = []
            next_tokens = []
            for en_len, next_token_pred, tokens in zip(target_lens, decoder_outputs, target_tensor):
                # e.g. Original sentence: <SOS> This is me . <EOS> length=en_len
                # Prediction: This is me . <EOS> length=en_len-1
                next_token_preds.append(next_token_pred[:en_len-1, :])
                next_tokens.append(tokens[1:en_len])
            next_token_preds = torch.cat(next_token_preds, dim=0)
            next_tokens = torch.cat(next_tokens, dim=0)
            loss = loss_fn(next_token_preds, next_tokens)
            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()
        
        # Evaluation
        encoder.eval()
        decoder.eval()
        tot_loss = 0.0
        tot_len = 0.0
        for start in tqdm(range(0, len(eval_data), config.batch_size)):

            # Encoder 
            batch_input = eval_data[start:min(start+config.batch_size, len(eval_data)), :] # (B, max_len * 2 + 2)
            encoder_input = batch_input[:, :config.max_len] # (B, max_len)
            encoder_seq_lens = batch_input[:, config.max_len] # (B, )
            encoder_outputs, (encoder_hn, encoder_cn) = encoder(encoder_input, encoder_seq_lens)
                # encoder_outputs (B, #longest_len, hidden_size)
                # encoder_hn (1, B, hidden_size)
                # encoder_cn (1, B, hidden_size)

            # Decoder
            target_tensor = batch_input[:, config.max_len + 1: -1] # (B, max_len)
            target_lens = batch_input[:, -1] # (B, )
            decoder_outputs = decoder(target_tensor, encoder_outputs, encoder_seq_lens, encoder_hn, encoder_cn) # (B, max_len, EN_VOCAB_SIZE)

            # Loss
            next_token_preds = []
            next_tokens = []
            for en_len, next_token_pred, tokens in zip(target_lens, decoder_outputs, target_tensor):
                # e.g. Original sentence: <SOS> This is me . <EOS> length=en_len
                # Prediction: This is me . <EOS> length=en_len-1
                next_token_preds.append(next_token_pred[:en_len-1, :])
                next_tokens.append(tokens[1:en_len])
            next_token_preds = torch.cat(next_token_preds, dim=0)
            next_tokens = torch.cat(next_tokens, dim=0)
            loss = loss_fn(next_token_preds, next_tokens)
            tot_loss += loss.item() * next_tokens.shape[0]
            tot_len += next_tokens.shape[0]
            
        avg_loss = tot_loss / tot_len
        ppl = np.exp(avg_loss)
        print(f"Epoch: {epoch} Loss: {avg_loss} PPL: {ppl}")
        if prev_eval_loss < avg_loss:
            print("> Early Stopping Triggered.")
            break
        prev_eval_loss = avg_loss

        # Sampling
        samples = torch.randint(0, len(eval_data), (config.eval_sample_num,))
        # Encoder 
        batch_input = eval_data[samples, :]
        encoder_input = batch_input[:, :config.max_len] # (B, max_len)
        encoder_seq_lens = batch_input[:, config.max_len] # (B, )
        encoder_outputs, (encoder_hn, encoder_cn) = encoder(encoder_input, encoder_seq_lens)
            # encoder_outputs (B, #longest_len, hidden_size)
            # encoder_hn (1, B, hidden_size)
            # encoder_cn (1, B, hidden_size)
        # Decoder
        target_tensor = batch_input[:, config.max_len + 1: -1] # (B, max_len)
        target_lens = batch_input[:, -1] # (B, )
        decoder_outputs = decoder(None, encoder_outputs, encoder_seq_lens, encoder_hn, encoder_cn) # (B, max_len)
        # Output
        for idx in range(config.eval_sample_num):
            jp_original = []
            en_original = []
            en_generated = []

            for i in range(encoder_seq_lens[idx]):
                jp_original.append(jp_idx_to_token[batch_input[idx, i].item()])
            for i in range(target_lens[idx]):
                en_original.append(en_idx_to_token[target_tensor[idx, i].item()])
            for i in range(config.max_len):
                en_generated.append(en_idx_to_token[decoder_outputs[idx, i].item()])
                if decoder_outputs[idx, i].item() == EOS_TOKEN:
                    break
            print(f"Sample {idx}: {jp_original} {en_original} {en_generated}")
    
    # Save model
    torch.save(encoder.state_dict(), "./model/encoder_params.pth")
    torch.save(decoder.state_dict(), "./model/decoder_params.pth")