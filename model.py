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
        for param in self.lm.parameters():
            param.requires_grad = False
        self.img_embedder = ImageEmbedding()
        self.mlp = nn.Sequential(
            nn.Linear(4864, 4*self.h_dim),
            nn.ReLU(),
            nn.Linear(4*self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, 2),
        )

        self.bn = torch.nn.BatchNorm1d(4864, momentum=.999, affine=False)

    def forward(self, img1, img2, sents, sent_lens):
        # [b, 512*2]
        img_features = self.img_embedder(img1, img2)

        
        # [b, s, 768]
        lm_last_hidden_states = self.lm(sents)[0]

        tot_embedding = torch.cat(
            [img_features, lm_last_hidden_states[:, 0]], dim=-1)
        

        tot_embedding = self.bn(tot_embedding)

        return self.mlp(tot_embedding)
