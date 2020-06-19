import torch
from torch.nn import functional as F, Parameter
import numpy as np
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


class Tucker(torch.nn.Module):
    def __init__(self, params):
        super(Tucker, self).__init__()
        self.name = 'Tucker'

        self.E = torch.nn.Embedding(params['num_entities'], params['embedding_dim'], padding_idx=0)
        self.R = torch.nn.Embedding(params['num_relations'], params['embedding_dim'], padding_idx=0)
        self.W = torch.nn.Parameter(torch.tensor(
            np.random.uniform(-1, 1, (params['embedding_dim'], params['embedding_dim'], params['embedding_dim'])),
            dtype=torch.float, requires_grad=True))

        self.input_dropout = torch.nn.Dropout(0.1)
        self.hidden_dropout1 = torch.nn.Dropout(0.1)
        self.hidden_dropout2 = torch.nn.Dropout(0.1)
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm1d(params['embedding_dim'])
        self.bn1 = torch.nn.BatchNorm1d(params['embedding_dim'])

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def forward(self, e1_idx, r_idx):
        e1 = self.E(e1_idx)
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        r = self.R(r_idx)
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat)
        x = x.view(-1, e1.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, self.E.weight.transpose(1, 0))
        pred = torch.sigmoid(x)
        return pred


class Hyper(torch.nn.Module):
    def __init__(self, params):
        super(Hyper, self).__init__()
        self.name = 'Hyper'
        self.in_channels = 1

        if 'conv_out' not in params:
            self.out_channels = 32
        else:
            self.out_channels = params["conv_out"]

        if 'filt_h' not in params:
            self.filt_h = 1

        if 'filt_w' not in params:
            self.filt_w = 9
        else:
            self.filt_w = params['filt_w']

        self.E = torch.nn.Embedding(params['num_entities'], params['embedding_dim'], padding_idx=0)
        self.R = torch.nn.Embedding(params['num_relations'], params['embedding_dim'], padding_idx=0)
        self.inp_drop = torch.nn.Dropout(0.2)
        self.hidden_drop = torch.nn.Dropout(0.3)
        self.feature_map_drop = torch.nn.Dropout2d(0.2)
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm2d(self.in_channels)
        self.bn1 = torch.nn.BatchNorm2d(self.out_channels)
        self.bn2 = torch.nn.BatchNorm1d(params['embedding_dim'])
        self.register_parameter('b', Parameter(torch.zeros(params['num_entities'])))
        fc_length = (1 - self.filt_h + 1) * (params['embedding_dim'] - self.filt_w + 1) * self.out_channels
        self.fc = torch.nn.Linear(fc_length, params['embedding_dim'])
        fc1_length = self.in_channels * self.out_channels * self.filt_h * self.filt_w
        self.fc1 = torch.nn.Linear(params['embedding_dim'], fc1_length)

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def forward(self, e1_idx, r_idx):
        e1 = self.E(e1_idx).view(-1, 1, 1, self.E.weight.size(1))
        r = self.R(r_idx)
        x = self.bn0(e1)
        x = self.inp_drop(x)

        k = self.fc1(r)
        k = k.view(-1, self.in_channels, self.out_channels, self.filt_h, self.filt_w)
        k = k.view(e1.size(0) * self.in_channels * self.out_channels, 1, self.filt_h, self.filt_w)

        x = x.permute(1, 0, 2, 3)

        x = F.conv2d(x, k, groups=e1.size(0))
        x = x.view(e1.size(0), 1, self.out_channels, 1 - self.filt_h + 1, e1.size(3) - self.filt_w + 1)
        x = x.permute(0, 3, 4, 1, 2)
        x = torch.sum(x, dim=3)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.bn1(x)
        x = self.feature_map_drop(x)
        x = x.view(e1.size(0), -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.E.weight.transpose(1, 0))
        x += self.b.expand_as(x)
        pred = torch.sigmoid(x)
        return pred


class Conve(torch.nn.Module):
    def __init__(self, params):
        super(Conve, self).__init__()
        self.name = 'Conve'
        self.emb_e = torch.nn.Embedding(params['num_entities'], params['embedding_dim'], padding_idx=0)
        self.emb_rel = torch.nn.Embedding(params['num_relations'], params['embedding_dim'], padding_idx=0)
        self.inp_drop = torch.nn.Dropout(params['input_dropout'])
        self.hidden_drop = torch.nn.Dropout(params['hidden_dropout'])
        self.feature_map_drop = torch.nn.Dropout2d(params['feature_map_dropout'])
        self.loss = torch.nn.BCELoss()

        self.emb_dim1 = params['embedding_dim'] // 5
        self.emb_dim2 = params['embedding_dim'] // self.emb_dim1

        self.conv1 = torch.nn.Conv2d(1, params['conv_out'], (3, 3), 1, 0, bias=True)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(params['conv_out'])
        self.bn2 = torch.nn.BatchNorm1d(params['embedding_dim'])
        self.register_parameter('b', Parameter(torch.zeros(params['num_entities'])))
        self.fc = torch.nn.Linear(params['projection_size'], params['embedding_dim'])

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_embedded = self.emb_e(e1).view(-1, 1, self.emb_dim1, self.emb_dim2)
        rel_embedded = self.emb_rel(rel).view(-1, 1, self.emb_dim1, self.emb_dim2)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.emb_e.weight.transpose(1, 0))
        x += self.b.expand_as(x)
        pred = torch.sigmoid(x)

        return pred


class Complex(torch.nn.Module):
    def __init__(self, params):
        super(Complex, self).__init__()
        self.name = 'Complex'
        self.num_entities = params['num_entities']
        self.num_relations = params['num_relations']
        self.embedding_dim = params['embedding_dim']

        self.emb_e_real = torch.nn.Embedding(self.num_entities, self.embedding_dim,
                                             padding_idx=0)
        self.emb_e_img = torch.nn.Embedding(self.num_entities, self.embedding_dim, padding_idx=0)
        self.emb_rel_real = torch.nn.Embedding(self.num_relations, self.embedding_dim, padding_idx=0)
        self.emb_rel_img = torch.nn.Embedding(self.num_relations, self.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(params['input_dropout'])
        self.loss = torch.nn.BCELoss()

        self.bn0_1 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn0_2 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn0_3 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn0_4 = torch.nn.BatchNorm1d(self.embedding_dim)

    def init(self):
        xavier_normal_(self.emb_e_real.weight.data)
        xavier_normal_(self.emb_e_img.weight.data)
        xavier_normal_(self.emb_rel_real.weight.data)
        xavier_normal_(self.emb_rel_img.weight.data)

    def forward(self, e1, rel):
        e1_embedded_real = self.emb_e_real(e1).squeeze()
        rel_embedded_real = self.emb_rel_real(rel).squeeze()
        e1_embedded_img = self.emb_e_img(e1).squeeze()
        rel_embedded_img = self.emb_rel_img(rel).squeeze()

        e1_embedded_real = self.bn0_1(self.inp_drop(e1_embedded_real))
        rel_embedded_real = self.bn0_2(self.inp_drop(rel_embedded_real))
        e1_embedded_img = self.bn0_3(self.inp_drop(e1_embedded_img))
        rel_embedded_img = self.bn0_4(self.inp_drop(rel_embedded_img))

        # complex space bilinear product (equivalent to HolE)
        realrealreal = torch.mm(e1_embedded_real * rel_embedded_real, self.emb_e_real.weight.transpose(1, 0))
        realimgimg = torch.mm(e1_embedded_real * rel_embedded_img, self.emb_e_img.weight.transpose(1, 0))
        imgrealimg = torch.mm(e1_embedded_img * rel_embedded_real, self.emb_e_img.weight.transpose(1, 0))
        imgimgreal = torch.mm(e1_embedded_img * rel_embedded_img, self.emb_e_real.weight.transpose(1, 0))
        pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        pred = torch.sigmoid(pred)

        return pred
