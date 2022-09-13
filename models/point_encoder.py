import torch
import torch.nn as nn

class PointEncoder(nn.Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, points_supervision, pos_encoder, label_encoder):
        batch_size = len(points_supervision)
        embeddings = []
        for idx in range(batch_size):
            position_embedding = pos_encoder.calc_emb(points_supervision[idx]['points'])
            label_embedding = label_encoder.calc_emb(points_supervision[idx]['labels'])
            embeddings.append(position_embedding + label_embedding)
        return embeddings

def build_point_encoder():
    return PointEncoder()
