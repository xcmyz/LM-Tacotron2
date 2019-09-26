import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pytorch_pretrained_bert import BertTokenizer
from multiprocessing import cpu_count
import math
import os

from text import text_to_sequence, sequence_to_text
import hparams

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def add_cls_sep(text):
    return "[CLS] " + text + " [SEP]"


def get_embedding(text, bert_model, tokenizer):
    s = text_to_sequence(text, hparams.text_cleaners)
    text_cleaned = sequence_to_text(s)

    text_processed = add_cls_sep(text_cleaned)
    tokenized_text = tokenizer.tokenize(text_processed)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0 for i in range(len(indexed_tokens))]
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    with torch.no_grad():
        encoded_layers, _ = bert_model(tokens_tensor, segments_tensors)

    bert_embeddings = encoded_layers[11][0]
    bert_embeddings = bert_embeddings[1:(bert_embeddings.size(0)-1)]

    text_list = tokenizer.tokenize(text_cleaned)
    index_list = list()
    text_out = ""
    cnt = 0
    for ele in text_list:
        if "##" != ele[0:2]:
            text_out += (ele+" ")
            index_list.append((cnt, cnt+len(ele)+1))
            cnt += (len(ele)+1)
        else:
            temp_word = ele[2:]
            text_out += (temp_word+" ")
            index_list.append((cnt, cnt+len(temp_word)+1))
            cnt += (len(temp_word)+1)

    embedding_list = list()
    for i, embedding in enumerate(bert_embeddings):
        embedding_list.append(embedding.expand(
            (index_list[i][1]-index_list[i][0]), -1))
    bert_embeddings = torch.cat(embedding_list, 0)

    return bert_embeddings


def get_clean_character(text, tokenizer):
    s = text_to_sequence(text, hparams.text_cleaners)
    text_cleaned = sequence_to_text(s)

    text_list = tokenizer.tokenize(text_cleaned)
    text_out = ""
    cnt = 0
    for ele in text_list:
        if "##" != ele[0:2]:
            text_out += (ele+" ")
            cnt += (len(ele)+1)
        else:
            temp_word = ele[2:]
            text_out += (temp_word+" ")
            cnt += (len(temp_word)+1)

    return text_out


class BERTTacotron2Dataset(Dataset):
    """ LJSpeech """

    def __init__(self, dataset_path=hparams.dataset_path):
        self.dataset_path = dataset_path
        self.text_path = os.path.join(self.dataset_path, "train.txt")
        self.text = process_text(self.text_path)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        index = idx + 1
        mel_name = os.path.join(
            self.dataset_path, "ljspeech-mel-%05d.npy" % index)
        mel_target = np.load(mel_name)

        character = self.text[idx][0:len(self.text[idx])-1]
        character = get_clean_character(character, self.tokenizer)
        character = np.array(text_to_sequence(
            character, hparams.text_cleaners))

        bert_embedding = np.load(os.path.join(
            hparams.bert_embeddings_path, str(idx)+".npy"))
        bert_embedding = torch.from_numpy(bert_embedding)

        stop_token = np.array([0. for _ in range(mel_target.shape[0])])
        stop_token[-1] = 1.

        sample = {"text": character, "mel_target": mel_target,
                  "bert_embedding": bert_embedding, "stop_token": stop_token}

        return sample


def process_text(train_text_path):
    with open(train_text_path, "r", encoding="utf-8") as f:
        txt = []
        for line in f.readlines():
            txt.append(line)

        return txt


def reprocess(batch, cut_list):
    texts = [batch[ind]["text"] for ind in cut_list]
    bert_embeddings = [batch[ind]["bert_embedding"] for ind in cut_list]
    mel_targets = [batch[ind]["mel_target"] for ind in cut_list]
    stop_tokens = [batch[ind]["stop_token"] for ind in cut_list]

    length_text = np.array([])
    for text in texts:
        length_text = np.append(length_text, text.shape[0])

    length_mel = np.array([])
    for mel in mel_targets:
        length_mel = np.append(length_mel, mel.shape[0])

    texts = pad_normal(texts)
    stop_tokens = pad_normal(stop_tokens, PAD=1.)
    mel_targets = pad_mel(mel_targets)
    bert_embeddings = pad_emb(bert_embeddings)

    out = {"text": texts, "mel_target": mel_targets, "stop_token": stop_tokens,
           "bert_embeddings": bert_embeddings, "length_mel": length_mel, "length_text": length_text}

    return out


def collate_fn(batch):
    len_arr = np.array([d["text"].shape[0] for d in batch])
    index_arr = np.argsort(-len_arr)
    batchsize = len(batch)
    real_batchsize = int(math.sqrt(batchsize))

    cut_list = list()
    for i in range(real_batchsize):
        cut_list.append(index_arr[i*real_batchsize:(i+1)*real_batchsize])

    output = list()
    for i in range(real_batchsize):
        output.append(reprocess(batch, cut_list[i]))

    return output


def pad_normal(inputs, PAD=0):

    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode='constant', constant_values=PAD)
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_mel(inputs):

    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(x, (0, max_len - np.shape(x)
                              [0]), mode='constant', constant_values=PAD)
        return x_padded[:, :s]

    max_len = max(np.shape(x)[0] for x in inputs)
    mel_output = np.stack([pad(x, max_len) for x in inputs])

    return mel_output


def pad_emb(inputs):

    def pad(x, max_len):
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")
        s = x.size(1)
        x_padded = F.pad(x, (0, 0, 0, max_len-x.size(0)))
        return x_padded[:, :s]

    max_len = max(x.size(0) for x in inputs)
    mel_output = torch.stack([pad(x, max_len) for x in inputs])

    return mel_output
