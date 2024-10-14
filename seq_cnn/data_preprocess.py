import jieba
import json

def load_data(split):
    data = []

    with open(f'{split}.txt', 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            text, label = line.rsplit('\t', 1)
            data.append([text, int(label)])
    
    return data

def construct_vocab(train_data):
    vocab_dict = {}
    vocab_size = 0

    for data in train_data:
        text = data[0]
        tokens = jieba.lcut(text, cut_all=False)
        for token in tokens:
            if token not in vocab_dict:
                vocab_dict[token] = vocab_size
                vocab_size += 1
    
    # If token in eval or test set is not included, replace with <UNK>
    vocab_dict["<UNK>"] = vocab_size
    vocab_size += 1

    return vocab_dict, vocab_size

if __name__ == "__main__":
    # Construct model vocabulary with training data
    train_data = load_data("train")
    vocab_dict, vocab_size = construct_vocab(train_data)
    with open("vocab.json", "w", encoding='utf-8') as file:
        json.dump(vocab_dict, file, ensure_ascii=True, indent=4)
    
    # Get tokenized train, eval, test data 
    splits = ['train', 'dev', 'test']
    for split in splits:
        tokenized_data = []
        data = load_data(split)
        for text, label in data:
            tokenized_seq = []
            tokens = jieba.lcut(text, cut_all=False)
            for token in tokens:
                if token in vocab_dict:
                    tokenized_seq.append(vocab_dict[token])
                else:
                    tokenized_seq.append(vocab_dict["<UNK>"])
            tokenized_data.append([tokenized_seq, label])
        with open(f"{split}.json", "w") as file:
            json.dump(tokenized_data, file)