import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import DistilBertModel

from img_embedding import ImageEmbedding

class DiffEval(nn.Module):
    def __init__(self):
        super(DiffEval, self).__init__()
        self.h_dim = 256
        self.lm = DistilBertModel.from_pretrained('distilbert-base-cased')
        self.img_embedder = ImageEmbedding(self.h_dim)
    
    def forward(self, img1, img2, sents, sent_lens):
        img_features = self.img_embedder(img1, img2)
        print(img_features.shape)
        lm_last_hidden_states = self.lm(sents)[0]
        print(lm_last_hidden_states.shape)
        return lm_last_hidden_states