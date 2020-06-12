from sklearn.base import BaseEstimator, TransformerMixin
from rdflib import Graph, URIRef, Namespace  # basic RDF handling


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

        self.kg_path+= '.nt'
        g.serialize(self.kg_path, format='ntriples')

        return g,self.kg_path,
