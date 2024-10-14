import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import MeCab
import numpy as np
from tqdm import tqdm
from utils.model import EncoderRNN, DecoderAttnRNN, RNNConfig
from utils.constants import EOS_TOKEN
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from nltk.translate.bleu_score import corpus_bleu
import pdb


def morph(tokenizer):
    def morphed_tokenizer(text):
        return tokenizer.parse(text).split()
    return morphed_tokenizer

class TrainingConfig():
    def __init__(self):
        self.batch_size = 64
        self.max_len = 72

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

def sampling(data, encoder, decoder, config=TrainingConfig()):

    # Vocab
    with open("./dataset/vocab_jp.json", 'r', encoding='utf-8') as file:
        jp_vocab_dict = json.load(file)
    with open("./dataset/vocab_en.json", 'r', encoding='utf-8') as file:
        en_vocab_dict = json.load(file)
    jp_idx_to_token = {value:key for key, value in jp_vocab_dict.items()}
    en_idx_to_token = {value:key for key, value in en_vocab_dict.items()}

    hypotheses = []
    references = []

    for start in range(0, len(data), config.batch_size):
        # Encoder 
        batch_input = data[start:min(start+config.batch_size, len(data)), :]
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
        for idx in range(batch_input.shape[0]):
            en_original = []
            en_generated = []

            # <SOS> and <EOS> ignored
            for i in range(1, target_lens[idx] - 1):
                en_original.append(en_idx_to_token[target_tensor[idx, i].item()])
            for i in range(config.max_len):
                if decoder_outputs[idx, i].item() == EOS_TOKEN:
                    break
                en_generated.append(en_idx_to_token[decoder_outputs[idx, i].item()])
            
            hypotheses.append(en_generated)
            references.append([en_original])
    
    return hypotheses, references

if __name__ == "__main__":

    # Dataset
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    SPLIT = 'test'
    test_data = data_proc(SPLIT, device)

    # Model
    encoder = EncoderRNN().to(device)
    decoder = DecoderAttnRNN().to(device)
    encoder.load_state_dict(torch.load("./model/encoder_params.pth"))
    decoder.load_state_dict(torch.load("./model/decoder_params.pth"))
    loss_fn = nn.CrossEntropyLoss()
    config = TrainingConfig()

    # Test (Loss, PPL, BLEU-4)
    encoder.eval()
    decoder.eval()
    tot_loss = 0.0
    tot_len = 0.0
    for start in tqdm(range(0, len(test_data), config.batch_size)):

        # Encoder 
        batch_input = test_data[start:min(start+config.batch_size, len(test_data)), :] # (B, max_len * 2 + 2)
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
    hypotheses, references = sampling(test_data, encoder, decoder)
    bleu_score = corpus_bleu(references, hypotheses)
    print(f"Loss: {avg_loss} PPL: {ppl} BLEU-4: {bleu_score}")