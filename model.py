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
        self.linear1 = nn.Linear(self.h_dim*4, self.h_dim*2)
        self.linear2 = nn.Linear(self.h_dim*2, self.h_dim)
        self.linear3 = nn.Linear(self.h_dim, 1)
    
    def forward(self, img1, img2, sents, sent_lens):
        img_features = self.img_embedder(img1, img2)
        print(img_features.shape)
        lm_last_hidden_states = self.lm(sents)[0]
        print(lm_last_hidden_states.shape)
        tot_embedding = torch.cat([img_features, lm_last_hidden_states[:,0]], dim=1)
        feature_layer1 = self.linear1(tot_embedding)
        feature_layer1 = nn.functional.relu(feature_layer1)
        feature_layer2 = self.linear2(feature_layer1)
        feature_layer2 = nn.functional.relu(feature_layer2)
        feature_layer3 = self.linear3(feature_layer2)
        feature_layer3 = nn.functional.relu(feature_layer3)
        feature_layer3 = torch.sigmoid(feature_layer3)
        return feature_layer3