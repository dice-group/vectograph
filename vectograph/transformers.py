from sklearn.base import BaseEstimator, TransformerMixin
from rdflib import Graph, URIRef, Namespace  # basic RDF handling
from .kge_models import DistMult
from .helper_classes import Data
import torch
import matplotlib.pyplot as plt
import numpy as np


class RDFGraphCreator(BaseEstimator, TransformerMixin):
    def __init__(self, path, dformat):

        self.kg_path = path
        self.kg_format = dformat

    def fit(self, x, y=None):
        """

        :param x:
        :param y:
        :return:
        """
        return self

    def transform(self, df):
        """

        :param df:
        :return:
        """
        print('Transformation starts')
        df.index = 'Event_' + df.index.astype(str)

        g = Graph()
        ppl = Namespace('http://dakiri.org/index/')
        schema = Namespace('http://schema.org/')

        for subject, row in df.iterrows():
            for predicate, obj in row.iteritems():
                if isinstance(obj, int):
                    g.add(
                        (URIRef(ppl + 'Event_' + subject), URIRef(schema + predicate), URIRef(ppl + 'num_' + str(obj))))
                elif isinstance(obj, float):
                    g.add((URIRef(ppl + 'Event_' + subject), URIRef(schema + predicate),
                           URIRef(ppl + 'float_' + str(obj))))
                elif isinstance(obj, str):
                    g.add(
                        (URIRef(ppl + 'Event_' + subject), URIRef(schema + predicate), URIRef(ppl + 'str_' + str(obj))))
                else:
                    raise ValueError

        self.kg_path += '.nt'
        g.serialize(self.kg_path, format='ntriples')

        return g, self.kg_path,


class KGCreator(BaseEstimator, TransformerMixin):
    """
    Direct convertion to txt file.
    """

    def __init__(self, path):
        self.kg_path = path
        self.model = None

    def fit(self, x, y=None):
        """
        :param x:
        :param y:
        :return:
        """
        return self

    def transform(self, df):
        """

        :param df:
        :return:
        """
        self.kg_path += '.txt'
        kg = []
        with open(self.kg_path, 'w') as writer:
            for subject, row in df.iterrows():
                for predicate, obj in row.iteritems():
                    triple = subject + '\t' + predicate + '\t' + str(obj) + '\n'
                    writer.write(triple)
                    kg.append(triple)

        return Data(kg=kg)


class ApplyKGE(BaseEstimator, TransformerMixin):
    def __init__(self, params):
        self.params = params

        # self.learning_rate = params['learning_rate'] do it later.
        self.model = None
        self.batch_size = params['batch_size']
        self.num_epochs = params['num_epochs']

    def fit(self, x, y=None):
        """
        :param x:
        :param y:
        :return:
        """
        return self

    def transform(self, data):
        """

        :param df:
        :return:
        """
        self.params['num_entities'] = len(data.entities)
        self.params['num_relations'] = len(data.relations)
        kge_name = self.params['kge']

        if kge_name == 'DistMult':
            self.model = DistMult(
                params={'num_entities': len(data.entities), 'embedding_dim': self.params['embedding_dim'],
                        'num_relations': len(data.relations), 'input_dropout': 0.2})
            self.model.init()

        print(self.model)
        print(self.params)

        train_data_idxs = data.get_data_idxs(data.triples)
        er_vocab = data.get_er_vocab(train_data_idxs)
        er_vocab_pairs = list(er_vocab.keys())

        opt = torch.optim.Adam(self.model.parameters())

        for it in range(1, self.num_epochs + 1):
            self.model.train()
            losses = []
            np.random.shuffle(er_vocab_pairs)
            for j in range(0, len(er_vocab_pairs), self.batch_size):
                data_batch, targets = data.get_batch(er_vocab, er_vocab_pairs, j,self.batch_size)
                opt.zero_grad()
                e1_idx = torch.tensor(data_batch[:, 0])
                r_idx = torch.tensor(data_batch[:, 1])
                predictions = self.model.forward(e1_idx, r_idx)
                loss = self.model.loss(predictions, targets)
                loss.backward()
                opt.step()
                losses.append(loss.item())

        plt.plot(losses)
        plt.show()