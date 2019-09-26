import torch
import numpy as np
from tqdm import tqdm
from pytorch_pretrained_bert import BertTokenizer, BertModel
import os

from data import ljspeech
from data_utils import get_embedding, process_text
import hparams as hp


def preprocess_ljspeech(filename):
    in_dir = filename
    out_dir = "dataset"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    metadata = ljspeech.build_from_path(in_dir, out_dir)
    write_metadata(metadata, out_dir)


def write_metadata(metadata, out_dir):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write(m + '\n')


def main():
    path = os.path.join("data", "LJSpeech-1.1")
    preprocess_ljspeech(path)

    model_bert = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    text_path = os.path.join(hp.dataset_path, "train.txt")
    texts = process_text(text_path)

    if not os.path.exists(hp.bert_embeddings_path):
        os.mkdir(hp.bert_embeddings_path)

    for ind, text in enumerate(texts):
        character = text[0:len(text)-1]
        bert_embedding = get_embedding(character, model_bert, tokenizer)
        np.save(os.path.join(hp.bert_embeddings_path, str(ind) + ".npy"),
                bert_embedding.numpy(), allow_pickle=False)

        if (ind+1) % 100 == 0:
            print("Done", (ind+1))


if __name__ == "__main__":
    main()
