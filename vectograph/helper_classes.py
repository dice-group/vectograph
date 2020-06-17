from collections import defaultdict
import numpy as np
import torch

class Data:

    def __init__(self, kg, reverse=False):

        self.triples = self.parse_data(kg, reverse=reverse)
        self.cuda=False
        del kg
        self.entities = self.get_entities(self.triples)
        self.tails = self.get_tails(self.triples)
        self.relations = self.get_relations(self.triples)

        self.entity_idxs = {self.entities[i]: i for i in range(len(self.entities))}
        self.relation_idxs = {self.relations[i]: i for i in range(len(self.relations))}
        self.train_data_idxs = self.get_data_idxs(self.triples)

    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], self.entity_idxs[data[i][2]]) for i
                     in range(len(data))]
        return data_idxs

    @staticmethod
    def parse_data(kg, reverse=False):
        data = [i.split() for i in kg]
        if reverse:
            data += ([i[2], i[1] + "_reverse", i[0]] for i in data)
        return data

    @staticmethod
    def get_relations(data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    @staticmethod
    def get_entities(data):
        entities = sorted(list(set([d[0] for d in data] + [d[2] for d in data])))
        return entities

    @staticmethod
    def get_tails(data):
        tails = sorted(list(set([d[2] for d in data])))
        return tails


    def get_er_vocab(self, data):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def get_batch(self, er_vocab, er_vocab_pairs, idx,batch_size):
        batch = er_vocab_pairs[idx:idx + batch_size]
        targets = np.zeros((len(batch), len(self.entities)))
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        targets = torch.FloatTensor(targets)
        if self.cuda:
            targets = targets.cuda()
        return np.array(batch), targets