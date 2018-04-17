"""
Author: Luca Soldaini (luca@soldaini.net)

Description: Script to test end-to-end system
    for relation classification/extraction
"""

# built-in modules
import os
import re
import shlex
import typing
import tempfile
import itertools
import subprocess
import collections

# installed modules
import spacy
import numpy as np
from bs4 import BeautifulSoup

# project modules
# None

# constants
INVALID_TAGS = ['text', 'title', 'abstract', 'body', 'html']


def load_abstracts_relations(
        subtask: typing.Union[float, int, str],
        load_test: bool=False
) -> typing.Tuple[list, list, list]:
    """
    Load abstracts, entities, and relations from dataset.

    :param subtask: Subtask to consider. Choose from 1.1, 1.2, or 2.
    :type subtask: typing.Union[float, int, str]

    :return: a tuple cointains:
        -   parsed_texts: a list of dictionaries, each of which containing
            the following keyed values:
                +   id: identifier for the document
                +   text: text of the doucment
        -   parsed_entities: a list of lists of entities in each
            document. Each list of entities associated with a document
            contains the following keyed values:
                +   id: entity identifier
                +   text: textual represenation of the entity
        -   parsed_relations: a list of lists of relations in each
            document. Each list of relations associated with a document
            contains the following keyed values:
                +   ent_a: entity involved in the relationship
                +   ent_b: entity involved in the relationship
                +   is_reverse: if false, relation is ent_a -> ent_b;
                    if true, relation is ent_b -> ent_a
                +   type: type of relationship
    :rtype: typing.Tuple[list, list, list]
    """

    if not load_test:
        # path to dataset containing abstracts and entities
        dataset_path = os.path.join('training-data', str(subtask), 'text.xml')
    else:
        dataset_path = os.path.join('test-data', str(subtask), 'text.xml')

    with open(dataset_path) as f:
        soup = BeautifulSoup(f, 'xml')

    parsed_entities = []
    parsed_texts = []

    # iterate over all titles/abstracts in the input text
    for title_abstract in soup.find_all('text'):

        text_id = title_abstract.attrs['id']

        parsed_passage = ''

        # add a full stop after the title of the paper,
        # so it can be merged with the abstract.
        title_abstract = BeautifulSoup(
            str(title_abstract).replace('</title>', '</title>.'),
            'lxml'
        )

        # remove tags arount title/abstract that we don't care about
        for tag in INVALID_TAGS:
            for match in title_abstract.findAll(tag):
                match.unwrap()

        # throw away all xml syntax, only keep the raw info for
        # easy parsing and removal of entities. we want to go from
        #   ... rhetorical and <entity id="L08-1459.34">syntactic</entity>
        #   <entity id="L08-1459.35">properties</entity> of parentheticals
        #   as well as the <entity id="L08-1459.36">decisions</entity> ...
        # to
        #   ... rhetorical and syntactic properties of parentheticals
        #   as well as the decisions ...
        # plus a list of entities.
        title_abstract = str(title_abstract)

        # remove unwanted spaces and add spaces when not set.
        title_abstract = re.sub(r'<entity ', r' <entity ', title_abstract)
        title_abstract = re.sub(r'>\s*', r'>', title_abstract)
        title_abstract = re.sub(r'\s*</entity>', r'</entity> ', title_abstract)
        title_abstract = re.sub(r'\s+', ' ', title_abstract).strip()
        parsed_entities.append([])

        i = 0   # moving pointer to character in the passsage
        while True:
            # keep looping until all entities have been extracted

            # find the start of the next entity
            next_entity = title_abstract.find('<entity', i)

            if next_entity < 0:
                # if the start is -1, there are no more entities to
                # extract, so add the last remaining bit of text to
                # parsed passage and get out of the while True loop.
                parsed_passage += title_abstract[i:]
                break

            # add the text between the last entity extacted and
            # the new entity to the parsed passage.
            parsed_passage += title_abstract[i:next_entity]

            # determine where the actual entity starts and ends, ignoring
            # the "<entity>" and "</entity>" tags.
            start_entity = title_abstract.find('">', next_entity) + 2
            end_entity = title_abstract.find('</entity>', start_entity)

            # extract the text of the entity
            entity_text = title_abstract[start_entity:end_entity]

            # extract the id of the entity
            entity_id = re.search(
                "\"(.+?)\"", title_abstract[next_entity:start_entity]
            ).groups()[0]

            # calculate where to start to scan next based on the
            # end entity offset and the length on the tag
            i = end_entity + len('</entity>')

            # derive position of the entity in the new parsed passage
            start_entity_in_parsed_passage = len(parsed_passage)
            parsed_passage += entity_text
            end_entity_in_parsed_passage = len(parsed_passage)

            # add entity to list of entitities
            parsed_entities[-1].append({
                'id': entity_id,
                'text': entity_text,
                'start': start_entity_in_parsed_passage,
                'end': end_entity_in_parsed_passage
            })

        # add parsed text to list of texts.
        parsed_texts.append({
            'id': text_id,
            'text': parsed_passage
        })

    if not load_test:
        # path to file containing relations
        relations_path = os.path.join(
            'training-data', str(subtask), 'relations.txt'
        )
    else:
        # path to file containing relations
        relations_path = os.path.join(
            'test-data', str(subtask), 'relations.txt'
        )

    # set up list to group relations by document
    docs_ids = {text['id']: i for i, text in enumerate(parsed_texts)}
    parsed_relations = [[] for _ in docs_ids]


    # get relations out
    # note that no relations are given for subtask 2 test set
    with open(relations_path) as f:
        for ln in f:

            # strip end characters
            ln = ln.strip()

            # skip empty lines
            if not ln:
                continue

            # data format:
            # RELATION_TYPE(<ENTITY>,<ENTITY>)
            # or
            # RELATION_TYPE(<ENTITY>,<ENTITY>,<REVERSE>)

            # if not load_test:
            # separate relation type from entities
            rel_type, rel_data = ln.strip(')').split('(')
            # else:
            # rel_type = None
            # rel_data = ln.strip(')').strip('(')

            # parse entities, reverse if avaliable
            try:
                ent_a, ent_b, is_reverse = rel_data.split(',')
            except ValueError:
                ent_a, ent_b = rel_data.split(',')
                is_reverse = False

            # use doc id to determine the position in parsed_relations list
            doc_id = ent_a.split('.')[0]

            # casting to prevent warning in PyCharm
            doc_pos = int(docs_ids[doc_id])

            parsed_relations[doc_pos].append({
                'type': rel_type,
                'ent_a': ent_a,
                'ent_b': ent_b,
                'is_reverse': is_reverse,
            })

    return parsed_texts, parsed_entities, parsed_relations


def split_dataset_into_sentences(
        texts: list,
        entities: list,
        relations: list,
        include_negative_samples=True
) -> list:
    """
    Reformat training data to have positive (and negative samples, if needed),
    and to be organized per sentence.

    :param texts: text of input abstracts
    :type texts: list

    :param entities: list of entites in document
    :type entities: list

    :param relations: list of relations in documents
    :type relations: list

    :param include_negative_samples: if true, negative samples
        are included in the set
    :type include_negative_samples: bool

    :return: list of positive (and negative samples)
        for relation classification
    :rtype: list
    """

    nlp = spacy.load('en')

    samples = []

    for text, doc_ents, doc_rels in zip(texts, entities, relations):

        sentences = [sent for sent in nlp(text['text']).sents]

        # map the character start position of tokens in
        # passage to each sentence / position in sentence
        token_starts = {
            pos: (sent_id, sent_pos)
            for pos, sent_id, sent_pos in itertools.chain(*(
                [(token.idx, s, t) for t, token in enumerate(sent)]
                for s, sent in enumerate(sentences)
            ))
        }

        # map the character end position of tokens in
        # passage to each sentence / position in sentence
        token_ends = {
            pos: (sent_id, sent_pos)
            for pos, sent_id, sent_pos in itertools.chain(*(
                [
                    (token.idx + len(token), s, t)
                    for t, token in enumerate(sent)
                ]
                for s, sent in enumerate(sentences)
            ))
        }

        # reorganize relations in dictionary for easier use
        dict_doc_rels = {}
        for rel in doc_rels:
            id_ent_a = rel['ent_a'].split('.')
            id_ent_a = id_ent_a[0], int(id_ent_a[1])

            id_ent_b = rel['ent_b'].split('.')
            id_ent_b = id_ent_b[0], int(id_ent_b[1])

            dict_doc_rels.setdefault(
                id_ent_a, {}
            )[id_ent_b] = rel['type'], rel['is_reverse']

        # reorganize entites in dictionary for easier use
        sents_ents = collections.defaultdict(list)
        for ent in doc_ents:
            ent_sent_id, ent_sent_pos_start = token_starts[ent['start']]
            _, ent_sent_pos_end = token_ends[ent['end']]

            # offset by 1 for easier indexing using
            # list slicing
            ent_sent_pos_end += 1

            ent_id = ent['id'].split('.')
            ent_id = ent_id[0], int(ent_id[1])

            sents_ents[ent_sent_id].append({
                'id': ent_id,
                'text': ent['text'],
                'sentence': ent_sent_id,
                'sent_start': ent_sent_pos_start,
                'sent_end': ent_sent_pos_end,
                'start': ent['start'],
                'end': ent['end']
            })

        for s, sent in enumerate(sentences):

            tokens = [token.text.lower() for token in sent]

            for ent_a, ent_b in itertools.combinations(sents_ents[s], 2):

                # swap if entity a occurs after entity b
                if ent_a['id'][1] > ent_b['id'][1]:
                    ent_a, ent_b = ent_b, ent_a

                rel_info = \
                    dict_doc_rels.get(ent_a['id'], {}).get(ent_b['id'], None)

                # relation exists between the two entities
                if rel_info:
                    rel_type, rel_reverse = rel_info
                    rel_reverse = 1 if rel_reverse else 0
                else:
                    rel_type, rel_reverse = 'NONE', 0

                if not include_negative_samples and rel_type == 'NONE':
                    # skip here if negative samples are not needed
                    continue

                samples.append({
                    'id': text['id'],
                    'tokens': tokens,
                    'spacy': sent,
                    'ent_a_start': ent_a['sent_start'],
                    'ent_a': '{}.{}'.format(*ent_a['id']),
                    'ent_a_end': ent_a['sent_end'],
                    'ent_b_start': ent_b['sent_start'],
                    'ent_b': '{}.{}'.format(*ent_b['id']),
                    'ent_b_end': ent_b['sent_end'],
                    'type': rel_type,
                    'is_reverse': rel_reverse,
                })

    print('[info] {:,} total samples'.format(len(samples)))

    return samples


def get_dependencies_map(dataset: tuple) -> dict:
    """Parse training """
    texts, _, _ = dataset
    nlp = spacy.load('en')

    dependencies_map = {'<UNK>': 0}

    for text in texts:
        parsed = nlp(text['text'])
        for token in parsed:
            dependencies_map.setdefault(token.dep_, len(dependencies_map))

    return dependencies_map


def get_part_of_speech_map(dataset: tuple) -> dict:
    """Parse training """
    texts, _, _ = dataset
    nlp = spacy.load('en')

    pos_map = {'<UNK>': 0}

    for text in texts:
        parsed = nlp(text['text'])
        for token in parsed:
            pos_map.setdefault(token.pos_, len(pos_map))

    return pos_map


def get_distribution_ent_length(sentences: list) -> typing.Tuple[float, float]:
    """Parse training """

    lengths = []

    for sent in sentences:
        lengths.append(sent['ent_a_end'] - sent['ent_a_start'])
        lengths.append(sent['ent_b_end'] - sent['ent_b_start'])

    return float(np.average(lengths)), float(np.std(lengths))


def split_train_test(
        subtask: typing.Union[float, int, str],
        texts: list,
        entities: list,
        relations: list
) -> typing.Tuple[tuple, tuple]:
    """
    Split list of texts, entities, and relations into
    training and test datasets.

    :param subtask: Subtask to consider. Choose from 1.1, 1.2, or 2.
    :type subtask: typing.Union[float, int, str]

    :param texts: text of input abstracts
    :type texts: list

    :param entities: list of entites in document
    :type entities: list

    :param relations: list of relations in documents
    :type relations: list

    :return: A tuple of (texts, entities, relations) tuples split
        between training and text data
    :rtype typing.Tuple[tuple, tuple]
    """
    texts_train, texts_test = [], []
    entities_train, entities_test = [], []
    relations_train, relations_test = [], []

    # load the
    test_doc_list_path = os.path.join(
        'training-data', str(subtask), 'test.txt'
    )
    with open(test_doc_list_path) as f:
        test_set = {ln.strip() for ln in f}

    for doc_text, doc_entities, doc_relations in\
            zip(texts, entities, relations):
        if doc_text['id'] in test_set:
            texts_test.append(doc_text)
            entities_test.append(doc_entities)
            relations_test.append(doc_relations)
        else:
            texts_train.append(doc_text)
            entities_train.append(doc_entities)
            relations_train.append(doc_relations)

    return (
        (texts_train, entities_train, relations_train),
        (texts_test, entities_test, relations_test)
    )


def split_train_test_sentences(
        subtask: typing.Union[float, int, str],
        samples: list,
) -> typing.Tuple[list, list]:
    """
    Split list of texts, entities, and relations into
    training and test datasets.

    :param subtask: Subtask to consider. Choose from 1.1, 1.2, or 2.
    :type subtask: typing.Union[float, int, str]

    :param samples: list of positive (and negative samples)
        for relation classification
    :type samples: list

    :return: Two lists of training and test samples.
    :rtype typing.Tuple[tuple, tuple]
    """

    test_doc_list_path = os.path.join(
        'training-data', str(subtask), 'test.txt'
    )
    with open(test_doc_list_path) as f:
        test_set = {ln.strip() for ln in f}

    train_samples, test_samples = [], []
    for sample in samples:
        if sample['id'] in test_set:
            if sample['type'] == 'NONE':
                continue
            test_samples.append(sample)
        else:
            train_samples.append(sample)

    return train_samples, test_samples


def _format_output(relations: list) -> str:
    """
    Convert a file with relations to a string that can be
    evaluated by eval.pl script.

    :param relations: relations file to parse
    :return:
    """

    if len(relations) == 0:
        return ''

    if type(relations[0]) is list:
        # relations are nested, so let's un-nest them
        relations = itertools.chain(*relations)

    output = []
    for rel in relations:

        if rel['type'] == 'NONE':
            continue

        rel_repr = (
            '{}({},{},REVERSE)' if rel['is_reverse'] else '{}({},{})'
        ).format(rel['type'], rel['ent_a'], rel['ent_b'])

        output.append(rel_repr)

    return '\n'.join(output)

def write_predictions_to_file(predictions: list, path: str):
    """
       Write predictions to file. Predictions should have the same
       format used in the training data, i.e. each relation should
       be a dictionary with the following keys:
           -   ent_a: entity involved in the relationship
           -   ent_b: entity involved in the relationship
           -   is_reverse: if false, relation is ent_a -> ent_b;
               if true, relation is ent_b -> ent_a
           -   type: type of relationship

       :param predictions: predicted labels by a given relation
           extraction method.
       :type predictions: list

       :param path: destination of predictions
       :type path: str

       """

    with open(path, 'w') as f_preds:
        f_preds.write(_format_output(predictions))


def evaluate(labels: list, predictions: list) -> str:
    """
    Evaluate predictions. Predictions should have the same format used
    in the training data, i.e. each relation should be a dictionary
    with the following keys:
        -   ent_a: entity involved in the relationship
        -   ent_b: entity involved in the relationship
        -   is_reverse: if false, relation is ent_a -> ent_b;
            if true, relation is ent_b -> ent_a
        -   type: type of relationship

    :param labels: labels from documents in the test set
    :type labels: list

    :param predictions: predicted labels by a given relation
        extraction method.
    :type predictions: list

    :return: the output of evaluation tool "eval.pl"
    :rtype: str
    """

    with tempfile.NamedTemporaryFile('w', delete=False) as f_preds, \
            tempfile.NamedTemporaryFile('w', delete=False) as f_labels:

        f_preds.write(_format_output(predictions))
        f_labels.write(_format_output(labels))

        pred_path, labels_path = f_preds.name, f_labels.name

    cmd = shlex.split('perl eval.pl {} {}'.format(pred_path, labels_path))
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    raw_output = proc.communicate()

    stdout, stderr = map(lambda e: e.decode('utf-8'), raw_output)
    os.remove(pred_path)
    os.remove(labels_path)

    if stderr.strip():
        raise RuntimeError(stderr.strip())

    return stdout
