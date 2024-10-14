import json
import random
import MeCab
from nltk.tokenize import word_tokenize
from utils.constants import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN
import pdb

def morph(tokenizer):
    def morphed_tokenizer(text):
        return tokenizer.parse(text).split()
    return morphed_tokenizer

def load_corpus():
    data = []

    with open(f'./dataset/eng_jpn.txt', 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            jp_text, en_text = line.rsplit('\t', 1)
            # Exclude unnecessary quotation marks wrapping English sentences
            if en_text[0] == '"' and en_text[-1] == '"':
                en_text = en_text[1: -1]
            data.append([jp_text, en_text])
    
    random.shuffle(data)

    return data

def construct_vocab(train_data):

    tokenizer_jp = morph(MeCab.Tagger("-Owakati"))
    tokenizer_en = word_tokenize

    vocab_dict_jp = {'<PAD>': PAD_TOKEN, '<EOS>': EOS_TOKEN, '<UNK>': UNK_TOKEN, '<SOS>': SOS_TOKEN}
    vocab_size_jp = 4
    vocab_dict_en = {'<PAD>': PAD_TOKEN, '<EOS>': EOS_TOKEN, '<UNK>': UNK_TOKEN, '<SOS>': SOS_TOKEN}
    vocab_size_en = 4

    for jp_text, en_text in train_data:

        jp_tokens = tokenizer_jp(jp_text)
        en_tokens = tokenizer_en(en_text)

        for token in jp_tokens:
            if token not in vocab_dict_jp:
                vocab_dict_jp[token] = vocab_size_jp
                vocab_size_jp += 1
        
        for token in en_tokens:
            if token not in vocab_dict_en:
                vocab_dict_en[token] = vocab_size_en
                vocab_size_en += 1

    return vocab_dict_jp, vocab_size_jp, vocab_dict_en, vocab_size_en

def preprocess_split(split, vocab_dict_jp, vocab_dict_en):

    tokenizer_jp = morph(MeCab.Tagger("-Owakati"))
    tokenizer_en = word_tokenize

    with open(f'./dataset/{split}.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    preprocessed_data = []
    for jp_text, en_text in data:

        # <SOS> token at start of sentence
        jp_preprocessed = [SOS_TOKEN]
        en_preprocessed = [SOS_TOKEN]

        jp_tokens = tokenizer_jp(jp_text)
        en_tokens = tokenizer_en(en_text)

        for token in jp_tokens:
            if token in vocab_dict_jp:
                jp_preprocessed.append(vocab_dict_jp[token])
            else:
                jp_preprocessed.append(vocab_dict_jp['<UNK>'])

        for token in en_tokens:
            if token in vocab_dict_en:
                en_preprocessed.append(vocab_dict_en[token])
            else:
                en_preprocessed.append(vocab_dict_en['<UNK>'])

        # <EOS> at end of sentence
        jp_preprocessed.append(EOS_TOKEN)
        en_preprocessed.append(EOS_TOKEN)

        preprocessed_data.append([jp_preprocessed, en_preprocessed])

    return preprocessed_data


if __name__ == "__main__":

    random.seed(8151)

    # Spliting the corpus into train, eval, test set (approximately 8:1:1)
    print("> Splitting corpus.")
    corpus = load_corpus()
    with open("./dataset/train.json", 'w', encoding='utf-8') as file:
        json.dump(corpus[0: (len(corpus) // 10) * 8], file, ensure_ascii=False, indent=4)
    with open("./dataset/eval.json", 'w', encoding='utf-8') as file:
        json.dump(corpus[(len(corpus) // 10) * 8: (len(corpus) // 10) * 9], file, ensure_ascii=False, indent=4)
    with open("./dataset/test.json", 'w', encoding='utf-8') as file:
        json.dump(corpus[(len(corpus) // 10) * 9: ], file, ensure_ascii=False, indent=4)
    
    # Construct jp and en vocabulary according to train set
    print("> Constructing vocabulary.")
    with open('./dataset/train.json', 'r', encoding='utf-8') as file:
        train_data = json.load(file)
    vocab_dict_jp, vocab_size_jp, vocab_dict_en, vocab_size_en = construct_vocab(train_data)
    print(f"> Japanese Vocab size: {vocab_size_jp}")
    print(f"> English Vocab size: {vocab_size_en}")
    with open("./dataset/vocab_jp.json", 'w', encoding='utf-8') as file:
        json.dump(vocab_dict_jp, file, ensure_ascii=False, indent=4)
    with open("./dataset/vocab_en.json", 'w', encoding='utf-8') as file:
        json.dump(vocab_dict_en, file, ensure_ascii=False, indent=4)

    # Process the splits with the established vocabulary
    print("> Preprocessing.")
    splits = ['train', 'eval', 'test']
    for split in splits:
        preprocessed_data = preprocess_split(split, vocab_dict_jp, vocab_dict_en)
        with open(f"./dataset/{split}_preprocessed.json", 'w') as file:
            json.dump(preprocessed_data, file, indent=4)