import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import DistilBertModel, DistilBertConfig

from img_embedding import ImageEmbedding


class DiffEval(nn.Module):
    def __init__(self):
        super(DiffEval, self).__init__()
        self.h_dim = 256
        configuration = DistilBertConfig(vocab_size=28996, output_hidden_states=True)
        self.lm = DistilBertModel.from_pretrained('distilbert-base-cased', config=configuration)
        self.img_embedder = ImageEmbedding()
        self.img_diff = nn.Sequential(
            nn.Linear(2048*2, 2*self.h_dim),
            nn.ReLU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(2*self.h_dim+768, 2*self.h_dim),
            nn.ReLU(),
            nn.Linear(2*self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, 2),
        )

    def forward(self, img1, img2, sents, sent_lens):
        self.lm.eval()
        for param in self.lm.parameters():
            param.requires_grad = False

        # [b, 2048*2]
        img_features = self.img_embedder(img1, img2)
        img_diff = self.img_diff(img_features)

        bert_output = self.lm(sents)
        # [b, s, 768]
        # lm_last_cls_hidden_states = bert_output[0][:, 0]
        pooled_last_two_layer_states = bert_output[1][-2].max(dim=1)[0]

        tot_embedding = torch.cat(
            [img_diff, pooled_last_two_layer_states], dim=-1)
        
        return self.mlp(tot_embedding)
