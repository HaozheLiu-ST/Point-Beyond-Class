
import torch
from torch import nn

class LabelEncoder(nn.Module):

    def __init__(self, num_feats, num_classes):
        super().__init__()

        self.label_embed = nn.Embedding(num_classes, num_feats)
        nn.init.uniform_(self.label_embed.weight)

    def forward(self, labels):
        emb = self.label_embed.weight[labels]
        return emb

    def calc_emb(self, labels):
        return self.forward(labels)


def build_label_encoder(num_feats, num_classes):
    return LabelEncoder(num_feats, num_classes)




