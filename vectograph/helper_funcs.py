import os
from rdflib import Graph, Literal, URIRef, Namespace  # basic RDF handling
from rdflib.namespace import XSD  # most common namespaces
import pandas as pd

def create_RDF_Graph(df):
    def save(X):
        # Takes too much
        # Very ineffective.
        g = Graph()
        ppl = Namespace('http://dakiri.org/index/')
        schema = Namespace('http://schema.org/')

        counter = 0
        for subject, row in X.iterrows():
            for predicate, obj in row.iteritems():
                if isinstance(obj, int):
                    # g.add((URIRef(ppl + 'Event_' + subject), URIRef(schema + predicate),
                    #       Literal(obj, datatype=XSD.integer)))
                    g.add(
                        (URIRef(ppl + 'Event_' + subject), URIRef(schema + predicate), URIRef(ppl + 'num_' + str(obj))))
                elif isinstance(obj, float):
                    # g.add((URIRef(ppl + subject), URIRef(schema + predicate), Literal(obj, datatype=XSD.float)))
                    g.add((URIRef(ppl + 'Event_' + subject), URIRef(schema + predicate),
                           URIRef(ppl + 'float_' + str(obj))))
                elif isinstance(obj, str):
                    # TODO understand whether it is a URI or not.
                    #                    g.add((URIRef(ppl + subject), URIRef(schema + predicate), Literal(obj, datatype=XSD.string)))
                    g.add(
                        (URIRef(ppl + 'Event_' + subject), URIRef(schema + predicate), URIRef(ppl + 'str_' + str(obj))))
                elif isinstance(obj, pd._libs.interval.Interval):
                    # g.add((URIRef(ppl + str(subject)), URIRef(schema + predicate), Literal('bin_' + str(obj))))

                    print(obj)

                    exit(1)
                    g.add(
                        (URIRef(ppl + 'Event_' + subject), URIRef(schema + predicate), URIRef(ppl + 'bin_' + str(obj))))
                else:
                    raise ValueError
            counter += 1

            if counter == 1000:
                break
        g.serialize(kg_path, format='ntriples')

    kg_path = path_of_folder + tabular_data_name + '.nt'
    df.index = 'Event_' + df.index.astype(str)
    save(df)
    return kg_path


def create_knowledge_graph(df: pd.DataFrame):
    g = Graph()
    ppl = Namespace('http://dakiri.org/index/')
    schema = Namespace('http://schema.org/')

    # This takes very long time.
    for subject, row in df.iterrows():
        for predicate, obj in row.iteritems():

            if isinstance(obj, int):
                g.add((URIRef(ppl + subject), URIRef(schema + predicate),
                       Literal(obj, datatype=XSD.integer)))
            elif isinstance(obj, float):
                g.add((URIRef(ppl + subject), URIRef(schema + predicate), Literal(obj, datatype=XSD.float)))
            elif isinstance(obj, str):
                # TODO understand whether it is a URI or not.
                g.add((URIRef(ppl + subject), URIRef(schema + predicate), (URIRef(ppl + obj))))
            else:
                raise ValueError
    g.serialize(path_of_folder + tabular_data_name + '.nt', format='ntriples')
    return path_of_folder + tabular_data_name + '.nt'

def apply_PYKE(t):
    g,path=t
    os.system("python PYKE/execute.py --K 20 --kg_path {0}".format(path))