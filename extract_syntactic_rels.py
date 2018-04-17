# built-in modules

# installed modules
import csv

import spacy
import spacy.symbols
import networkx as nx

# project modules
import pipeline


def customize_tokenizer(spacy_pipeline):
    spacy_pipeline.tokenizer.add_special_case(
        'viz.', [{
            spacy.symbols.ORTH: 'namely',
            spacy.symbols.LEMMA: 'namely',
            spacy.symbols.POS: 'ADV'
        }]
    )


def parse_tree_to_graph(document):
    # Load spacy's dependency tree into a networkx graph
    graph = nx.DiGraph()

    for token in document:
        for child in token.children:
            graph.add_edge(
                token.i,
                child.i,
                label=child.dep_,
                direction=1,
                ent_a=token,
                ent_b=child
            )
            graph.add_edge(
                child.i,
                token.i,
                label=child.dep_,
                direction=-1,
                ent_a=child,
                ent_b=token
            )

    return graph


def get_distance_between_entities(ent_a, ent_b, graph):
    return nx.shortest_path_length(graph, ent_a, ent_b)


def get_nodes_between_entities(ent_a, ent_b, graph):
    return nx.shortest_path(graph, ent_a, ent_b)


def get_edges_between_entites(ent_a, ent_b, graph, include_terms=False):

    path = nx.shortest_path(graph, ent_a, ent_b)

    if include_terms:
        edges = [
            (
                graph[s][t]['label'],
                graph[s][t]['direction'],
                (graph[s][t]['ent_a'], graph[s][t]['ent_b'])
            )
            for s, t in zip(path, path[1:])
        ]
    else:
        edges = [
            (graph[s][t]['label'], graph[s][t]['direction'])
            for s, t in zip(path, path[1:])
        ]

    return edges


def get_entity_head(entity_tokens, graph):
    if len(entity_tokens) < 2:
        return entity_tokens[0].i

    for token in entity_tokens:
        # get direction of all edges from s
        edges_dirs = [
            graph[token.i][target.i]['direction']
            for target in entity_tokens
            if target.i in graph[token.i]
        ]

        # check if all other terms in the entity
        # has an edge to are children
        if all(map(lambda e: e == 1, edges_dirs)):
            return token.i


def load_ontology(path):
    graph = nx.DiGraph()

    with open(path) as f:
        rd = csv.reader(f)
        for k1, k2 in rd:
            graph.add_edge()


if __name__ == '__main__':
    subtask = '1.1'
    oth_path = '/home/ls/blue-hd/datasets/saffron-hierarchies-acl/saffron-ACL-cleaned.csv'

    nlp = spacy.load('en')
    customize_tokenizer(nlp)

    dataset = pipeline.load_abstracts_relations(subtask)

    # get list of all dependency tags used in the dataset
    dependencies_map = pipeline.get_dependencies_map(dataset)

    # get list of all pos tags used in the dataset
    pos_map = pipeline.get_part_of_speech_map(dataset)

    # split it by sentence, potentially include negative samples
    sentences_dataset = pipeline.split_dataset_into_sentences(*dataset)

    # split sentences between train and test according to the
    # official dataset split
    train_sentences, _ = pipeline.split_train_test_sentences(
        subtask, sentences_dataset)

    ontology = load_ontology(oth_path)
