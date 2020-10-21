import torch
from torch.nn.init import xavier_normal_


class Distmult(torch.nn.Module):
    def __init__(self, params):
        super(Distmult, self).__init__()
        self.name = 'Distmult'
        self.emb_e = torch.nn.Embedding(params['num_entities'], params['embedding_dim'], padding_idx=0)
        self.emb_rel = torch.nn.Embedding(params['num_relations'], params['embedding_dim'], padding_idx=0)
        self.inp_drop = torch.nn.Dropout(params['input_dropout'])
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm1d(params['embedding_dim'])
        self.bn1 = torch.nn.BatchNorm1d(params['embedding_dim'])

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_embedded = self.emb_e(e1)
        rel_embedded = self.emb_rel(rel)
        e1_embedded = e1_embedded.squeeze()
        rel_embedded = rel_embedded.squeeze()

        e1_embedded = self.bn0(self.inp_drop(e1_embedded))
        rel_embedded = self.bn1(self.inp_drop(rel_embedded))

        pred = torch.mm(e1_embedded * rel_embedded, self.emb_e.weight.transpose(1, 0))
        pred = torch.sigmoid(pred)

        return pred