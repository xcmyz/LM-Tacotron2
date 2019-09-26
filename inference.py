import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
from pytorch_pretrained_bert import BertTokenizer, BertModel
import os
import numpy as np

import waveglow
import glow
from network import BERT_Tacotron2
from text import text_to_sequence
import data_utils
import hparams as hp
import Audio

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_data(data, figsize=(12, 4)):
    _, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto',
                       origin='bottom', interpolation='none')

    if not os.path.exists("img"):
        os.mkdir("img")
    plt.savefig(os.path.join("img", "model_test.jpg"))


def get_model(num):
    checkpoint_path = "checkpoint_" + str(num) + ".pth.tar"
    model = nn.DataParallel(BERT_Tacotron2(hp)).to(device)
    model.load_state_dict(torch.load(os.path.join(
        hp.checkpoint_path, checkpoint_path))['model'])
    model.eval()

    return model


def synthesis(model, text, embeddings):
    with torch.no_grad():
        sequence = torch.autograd.Variable(
            torch.from_numpy(text)).cuda().long()

        mel_outputs, mel_outputs_postnet, _, alignments = model.module.inference(
            sequence, embeddings)

        return mel_outputs[0].cpu(), mel_outputs_postnet[0].cpu(), mel_outputs_postnet


if __name__ == "__main__":
    # Test
    num = 72000
    model = get_model(num)
    model_bert = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    words = "President Trump met with other leaders at the Group of 20 conference."
    embeddings = data_utils.get_embedding(words, model_bert, tokenizer)
    text = data_utils.get_clean_character(words, tokenizer)
    text = np.array(text_to_sequence(text, hp.text_cleaners))
    text = np.stack([text])
    embeddings = torch.stack([embeddings]).to(device)

    mel, mel_postnet, mel_postnet_torch = synthesis(model, text, embeddings)
    if not os.path.exists("results"):
        os.mkdir("results")
    Audio.tools.inv_mel_spec(mel_postnet, os.path.join(
        "results", words + str(num) + "griffin_lim.wav"))
    plot_data([mel.numpy(), mel_postnet.numpy()])

    waveglow_path = os.path.join("waveglow", "pre_trained_model")
    waveglow_path = os.path.join(waveglow_path, "waveglow_256channels.pt")
    wave_glow = torch.load(waveglow_path)['model']
    wave_glow = wave_glow.remove_weightnorm(wave_glow)
    wave_glow.cuda().eval()
    for m in wave_glow.modules():
        if 'Conv' in str(type(m)):
            setattr(m, 'padding_mode', 'zeros')
    waveglow.inference.inference(mel_postnet_torch, wave_glow, os.path.join(
        "results", words + str(num) + "waveglow.wav"))
