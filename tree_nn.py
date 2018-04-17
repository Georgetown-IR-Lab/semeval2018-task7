# built in modules
import random
import argparse
import typing
import itertools
import os
import sys
import zipfile

# installed modules
import tqdm
import mxnet as mx
import numpy as np
import mxnet.autograd

# project modules
import pipeline
from extract_syntactic_rels import (
    parse_tree_to_graph,
    get_nodes_between_entities,
    get_entity_head,
    get_edges_between_entites)
from utils.spacy_utils import get_token_root_distance
from utils import mxnet_utils
from utils.webutils import download_file_from_google_drive

# constants
EMBEDDINGS_PATHS = {
    'wiki': '~/green-hd/fasttext/wiki-news-300d-1M.txt',
    'arxiv': '~/green-hd/scientific-extraction/arxiv.cs.w8.d100.txt',
    'both': '~/green-hd/merged-wiki-arxiv.txt'
}
# LABELS_MAP = {
#     'NONE': 0,
#     'COMPARE': 1,
#     'COMPARE:REVERSE': 2,
#     'MODEL-FEATURE': 3,
#     'MODEL-FEATURE:REVERSE': 4,
#     'PART_WHOLE': 5,
#     'PART_WHOLE:REVERSE': 6,
#     'RESULT': 7,
#     'RESULT:REVERSE': 8,
#     'TOPIC': 9,
#     'TOPIC:REVERSE': 10,
#     'USAGE': 11,
#     'USAGE:REVERSE': 12
# }
CACHEDIR = 'cache'
LABELS_MAP = {
    'NONE': 0,
    'COMPARE': 1,
    'MODEL-FEATURE': 2,
    'PART_WHOLE': 3,
    'RESULT': 4,
    'TOPIC': 5,
    'USAGE': 6,
}
EMBEDDINGS_URLS = {
    'wiki': '10tzFRTo2_aAP_zx6QKyp_s8klr9t9op3',
    'arxiv': '17XxsRdDMAVIXT5hgJnCuzFI6nyQiSLts',
    'both': '1XM5WAQlVJ7CDfdG9jcYk2ao6jTHFfg4T'
}



def json_dict(text):
    try:
        parsed = json.loads(text)
        if type(parsed) is list:
            raise ValueError('Expected a Dictionary, not a list.')
    except (json.JSONDecodeError, ValueError) as e:
        raise argparse.ArgumentTypeError(e)

    return parsed



def download_embeddings(path, emb_type, cachedir=CACHEDIR):
    """Download embeddings from google drive"""

    if not os.path.exists(cachedir):
        os.mkdir(cachedir)

    if not os.path.exists(path):
        path_to_zip_file = os.path.join(cachedir, 'temp-{}.zip'.format(emb_type))
        print('[info] downloading embeddings of type {}...' end=' ')
        download_file_from_google_drive(EMBEDDINGS_URLS[emb_type], path_to_zip_file)
        with zipfile.ZipFile(path_to_zip_file, 'r') as zf:
            path = os.path.join(cachedir, zf.filename().replece('.zip', ''))
            zf.extractall('.')
        os.remove(path_to_zip_file)

    return path



class SyntacticTree:
    def __init__(self, idx):
        self.children = []
        self.edges = []
        self.idx = idx
        self.parent = None

    def add_child(self, child, edge=None):
        self.children.append(child)
        self.edges.append(edge)

    def __repr__(self):
        output = '{} '.format(self.idx)

        for child, edge in zip(self.children, self.edges):
            output += '({}-{}) '.format(edge, str(child))

        return output.strip()

    def to_array(self, total=0):
        if total == 0:
            root = self
            while True:
                if root.parent is None:
                    break
                else:
                    root = root.parent
            total = len(root)

        if self.children:
            children_ids = sum(
                child.to_array(total=total)
                for child in self.children
            )
        else:
            children_ids = np.zeros(shape=(total, total))

        for child in self.children:
            children_ids[self.idx, child.idx] = 1.

        return children_ids

    def __len__(self):
        return 1 + sum(len(child) for child in self.children)

    @classmethod
    def _split_string(cls, obj_repr):
        splits = []

        par_count = 0

        while len(obj_repr) > 0:
            for i, ch in enumerate(obj_repr):
                if ch == '(':
                    par_count += 1
                elif ch == ')':
                    par_count -= 1

                if ch == ')' and par_count == 0:
                    prev, obj_repr = obj_repr[1:i], obj_repr[i + 1:].strip()
                    splits.append(prev.strip())
                    break

        return splits

    @classmethod
    def load(cls, obj_repr):
        try:
            idx, rest = obj_repr.split(' ', 1)
        except:
            return cls(int(obj_repr))

        tree = cls(int(idx))

        for str_child in cls._split_string(rest):
            try:
                edge, str_child = str_child.split('-', 1)
            except ValueError:
                edge = None

            tree.add_child(cls.load(str_child), edge)

        return tree


def make_syntactic_tree(tokens):
    tokens = sorted(tokens, key=lambda t: t.i)
    tokens_map = dict(enumerate(tokens))
    id_map = {t.i: i for i, t in enumerate(tokens)}

    tree = {}
    for i, token in tokens_map.items():
        node = tree.setdefault(i, SyntacticTree(i))
        for child_token in token.children:

            if id_map.get(child_token.i, -1) not in tokens_map:
                continue

            child_node = tree.setdefault(
                id_map[child_token.i], SyntacticTree(id_map[child_token.i])
            )
            child_node.parent = node
            node.add_child(child_node, child_token.dep_)


    root = [n for n in tree.values() if n.parent is None][0]

    return root


class Net(mx.gluon.Block):
    def __init__(
        self,
        embedding_weights,
        output_classes_num,
        dropout=0.0,
        rnn_hidden_size=None,
        dense_hidden_size=None,
        trainable_embeddings=False,
        dependencies_num=0,
        part_of_speech_num=0,
        include_ent_len=False,
        max_tree_height=0,
        **kwargs
    ):
        super(Net, self).__init__(**kwargs)

        self.max_tree_height = max_tree_height
        self.dependencies_num  = dependencies_num
        self.part_of_speech_num = part_of_speech_num
        self.include_ent_len = include_ent_len

        if rnn_hidden_size is None:
            rnn_hidden_size = embedding_weights.shape[1] // 2

        if dense_hidden_size is None:
            dense_hidden_size = rnn_hidden_size // 2

        total_input_size = (
            embedding_weights.shape[1] +
            self.dependencies_num +
            self.part_of_speech_num +
            (1 if self.max_tree_height > 0 else 0) +
            (2 if self.include_ent_len else 0)
        )

        with self.name_scope():
            self.dropout = mx.gluon.nn.Dropout(dropout)

            self.embeddings = mxnet_utils.get_embedding_layer(
                embedding_weights=embedding_weights,
                is_trainable=trainable_embeddings
            )

            self.childsum_lstm = mxnet_utils.TreeLSTM(
                hidden_size=rnn_hidden_size,
                input_size=total_input_size,
                dropout=dropout
            )

            self.dense0 = mx.gluon.nn.Dense(
                units=dense_hidden_size,
                activation='relu',
            )
            self.dense1 = mx.gluon.nn.Dense(
                units=output_classes_num,
            )

    def forward(
            self,
            tokens, # tokens in syntactic tree
            deps, # labels for dependencies
            pos, # labels for part of speech tags
            ent_lens, # lengths of the two entities
            dist_from_tree, # distance of each token from tree head
            adj, # adjcency matrix of edges in syntactic tree
            entities, # position of entities in dependencies
            idx, # tree root
            is_training: bool=False # are we training?
    ) -> tuple:

        # tokens, adj, entities, idx, True

        forward_ctx = (
            mx.autograd.train_mode()
            if is_training
            else mx.autograd.predict_mode()
        )

        with forward_ctx:
            seq_embeddings = self.embeddings(tokens)


            dropped_seq_embeddings = self.dropout(seq_embeddings)

            if self.dependencies_num > 0:
                # if dependency tag features are enabled, concatenate
                # them to input embeddings
                seq_embeddings_dep = mx.nd.one_hot(deps, self.dependencies_num)
                dropped_seq_embeddings = mx.nd.concat(
                    dropped_seq_embeddings, seq_embeddings_dep, dim=1
                )

            if self.part_of_speech_num > 0:
                # if part of speech tag features are enabled, concatenate
                # them to input embeddings
                seq_embeddings_pos = mx.nd.one_hot(pos, self.part_of_speech_num)
                dropped_seq_embeddings = mx.nd.concat(
                    dropped_seq_embeddings, seq_embeddings_pos, dim=1
                )

            if self.max_tree_height > 0:
                dist_from_tree = mx.nd.divide(
                    dist_from_tree.reshape((-1, 1)),
                    self.max_tree_height
                )
                dropped_seq_embeddings = mx.nd.concat(
                    dropped_seq_embeddings, dist_from_tree, dim=1
                )

            if self.include_ent_len:
                dropped_seq_embeddings = mx.nd.concat(
                    dropped_seq_embeddings, ent_lens, dim=1
                )


            rnn_output, _ = self.childsum_lstm(dropped_seq_embeddings, adj, idx)
            dropped_rnn_output = self.dropout(rnn_output)

            dropped_rnn_output = dropped_rnn_output.reshape((1, -1))

            hidden = self.dense0(dropped_rnn_output)
            dropped_hidden = self.dropout(hidden)
            output = self.dense1(dropped_hidden)

        return output

    def initialize(self, init=mx.init.Xavier(), ctx=None, verbose=False):
        # re-inplemented so that Xavier is used instead of uniform
        super(Net, self).initialize(init, ctx, verbose)


def entity_children_iterator(entity):
    for token in entity:
        yield from token.children


def prepare_data_for_net(
        vocabulary: dict,
        samples: list,
        labels_map: dict,
        pos_map: dict,
        dependencies_map: dict,
        entity_length_distribution: typing.Tuple[float, float],
        include_entities_nodes: bool=False,
        include_entities_children: bool=False,
        case_sensitive: bool=False,
) -> typing.List[typing.Tuple[tuple, tuple, int, int]]:
    """
    Convert training data to ids in a way that so that it
    can be batched and fed into a RNN.

    :param vocabulary: vocabulary that maps strings to
        embedding ids
    :type vocabulary: dict

    :param samples: list of [training|test] samples to
        convert to numeric format
    :type samples: list

    :param labels_map: map of ids to use for labels
    :type labels_map: dict

    :return: iterator with training samples
    :rtype: typing.List[typing.Tuple[tuple, tuple, int, int]]
    """
    data = []

    def normalize_length(l):
        v = 1 / (1 + np.exp((entity_length_distribution[0] - l) /
                            entity_length_distribution[1]))
        return 2 * v - 1

    for sample in samples:
        parsed = sample['spacy']

        sentence_graph = parse_tree_to_graph(parsed)
        trueloc= lambda token: token.i - parsed[0].i

        span_ent_a = parsed[sample['ent_a_start']:sample['ent_a_end']]
        head_ent_a = get_entity_head(span_ent_a, sentence_graph)

        span_ent_b = parsed[sample['ent_b_start']:sample['ent_b_end']]
        head_ent_b = get_entity_head(span_ent_b, sentence_graph)

        offset = parsed[0].i
        in_between_nodes = [
            parsed[i - offset]
            for i in get_nodes_between_entities(
                head_ent_a, head_ent_b, sentence_graph
            )
        ]

        if include_entities_nodes:
            nodes_in_subgraph = set(
                itertools.chain(*(span_ent_a, span_ent_b, in_between_nodes))
            )
        else:
            nodes_in_subgraph = set(in_between_nodes)

        # saving it here before I add random children
        len_nodes_in_subgraph = len(nodes_in_subgraph)

        if include_entities_children:
            for tok in itertools.chain(*(
                    entity_children_iterator(node)
                    for node in nodes_in_subgraph
            )):
                nodes_in_subgraph.add(tok)

        sorted_nodes_in_subgraph = sorted(
            nodes_in_subgraph, key=lambda token: token.i)

        syntactic_subtree = make_syntactic_tree(sorted_nodes_in_subgraph)

        if len(syntactic_subtree) < len_nodes_in_subgraph:
            msg = (
                '[warning] skipping "{}": parser could not build '
                'syntactic tree\n'
                ''.format(' '.join(t.text for t in sorted_nodes_in_subgraph))
            )
            sys.stderr.write(msg)
            continue

        entities = tuple(
            1 if (
                sample['ent_a_start'] <= trueloc(token)< sample['ent_a_end'] or
                sample['ent_b_start'] <= trueloc(token) < sample['ent_b_end']
            ) else 0 for token in nodes_in_subgraph
        )

        label = sample['type'] #+ (':REVERSE' if sample['is_reverse'] else '')

        # if the label is not assigned to this sample
        # (as it is on the evaluation data) we simply ignore this
        # and set the label to -1 instead. (-1 would cause issue if
        # feed into the network, so it is a good error check).
        label_id = labels_map[label] if label is not None else -1

        syntactic_subtree_tokens = [
            vocabulary.get(
                (token.text.lower() if not case_sensitive else token.text),
                0
            ) for token in sorted_nodes_in_subgraph
        ]

        distance_memoizaiton_map = {}
        syntactic_subtree_absolute_distances = [
            get_token_root_distance(t, distance_memoizaiton_map)
            for t in sorted_nodes_in_subgraph
        ]
        mininum_distance_from_root = min(syntactic_subtree_absolute_distances)
        syntactic_subtree_relative_distances = [
            dist - mininum_distance_from_root
            for dist in syntactic_subtree_absolute_distances
        ]

        syntactic_subtree_dependencies = [
            dependencies_map.get(token.dep_, 0)
            for token in sorted_nodes_in_subgraph
        ]
        syntactic_subtree_pos = [
            pos_map.get(token.pos_, 0)
            for token in sorted_nodes_in_subgraph
        ]

        entities_length = [
            [
                normalize_length(sample['ent_a_end'] - sample['ent_a_start']),
                normalize_length(sample['ent_b_end'] - sample['ent_b_start']),
            ] for _ in sorted_nodes_in_subgraph
        ]

        # keep the small one before the large one
        pos_normed_entities_length = [
            [np.min(e), np.max(e)]
            for e in entities_length
        ]

        data.append((
            syntactic_subtree_tokens,
            syntactic_subtree_dependencies,
            syntactic_subtree_pos,
            pos_normed_entities_length,
            syntactic_subtree_relative_distances,
            syntactic_subtree,
            entities,
            label_id
        ))

    return data


def evaluate_on_test_data(
        net, test_sentences, test_data, labels_map,
        output_for_error_analysis=None,
        evaluate_output=None
):
    inverse_labels_map = {v: k for k, v in labels_map.items()}

    predictions = []

    if output_for_error_analysis:
        f = open(output_for_error_analysis, 'w')
    else:
        f = None

    for sample, sentence in zip(test_data, test_sentences):
        (
            tokens,             # the tokens in sentence
            deps,               # dependency tags
            pos,                # part of speech tags
            ent_lens,           # length of the input entities
            dist_from_tree,     # distance from root of subtree
            tree,               # the subtree
            entities,           # indication for entity location
            label               # the label for this sample
        ) = sample

        tokens = mx.nd.array(tokens)
        entities = mx.nd.array(entities)
        deps = mx.nd.array(deps)
        idx = mx.nd.array([tree.idx])
        adj = mx.nd.array(tree.to_array())
        pos = mx.nd.array(pos)
        ent_lens = mx.nd.array(ent_lens)
        dist_from_tree = mx.nd.array(dist_from_tree)

        prob = net(
            tokens, deps, pos, ent_lens,
            dist_from_tree, adj, entities, idx,
            False   # is testing
        )

        pred_class_id = int(mx.nd.argmax(prob, axis=1).asscalar())
        pred_class = inverse_labels_map[pred_class_id]
        pred_prob = mx.nd.softmax(prob, axis=1).reshape((-1, ))

        # if pred_class == 'NONE':
        #     continue

        is_reversed = ':' in pred_class
        pred_class = pred_class.split(':')[0]

        predictions.append({
            'ent_a': sentence['ent_a'],
            'ent_b': sentence['ent_b'],
            # 'is_reverse': is_reversed,
            'is_reverse': sentence['is_reverse'],
            'type': pred_class,
        })

        if f:
            tokens = sentence['spacy']

            graph = parse_tree_to_graph(tokens)

            ent_a = tokens[sentence['ent_a_start']:sentence['ent_a_end']]
            ent_a_head_id = get_entity_head(ent_a, graph)
            ent_a_head = tokens.doc[ent_a_head_id]

            ent_b = tokens[sentence['ent_b_start']:sentence['ent_b_end']]
            ent_b_head_id = get_entity_head(ent_b, graph)
            ent_b_head = tokens.doc[ent_b_head_id]

            f.write('sentence: "{}"\n'.format(tokens))
            f.write('entity_a: "{}" (head: "{}")\n'.format(ent_a, ent_a_head))
            f.write('entity_b: "{}" (head: "{}")\n'.format(ent_b, ent_b_head))
            f.write('tree: {}\n'.format(str(ent_a_head) + ' ' + ' '.join(
                ('-> {}' if direction > 0 else '<- {}').format(tb)
                for _, direction, (_, tb) in get_edges_between_entites(
                    ent_a_head_id, ent_b_head_id, graph, include_terms=True
                )
            )))
            f.write('relation: {}{}\n'.format(
                sentence['type'],
                '-reversed' if sentence['is_reverse'] else ''))
            f.write('predicted: {}{}\n'.format(
                pred_class, '-reversed' if is_reversed else ''
            ))
            f.write('confidence: {:.2%}\n'.format(
                pred_prob[pred_class_id].asscalar())
            )
            f.write('\n\n')

    if f:
        f.close()


    if evaluate_output:
        pipeline.write_predictions_to_file(
            predictions=predictions, path=evaluate_output)
    else:
        resp = pipeline.evaluate(
            predictions=predictions, labels=test_sentences)
        print(resp)


def main(opts):
    # set a seed for reproducible network
    random.seed(42)

    labels_map = opts.labels_map
    if not labels_map:
        # get a copy of the labels if not provided
        labels_map = dict(LABELS_MAP)

    if not opts.include_negative_samples:
        # pop out the "NONE" label if no negative
        # samples are provided
        labels_map.pop('NONE')
        labels_map = {k: v - 1 for k, v in labels_map.items()}

    # load the dataset
    dataset = pipeline.load_abstracts_relations(opts.subtask)

    # get list of all dependency tags used in the dataset
    dependencies_map = pipeline.get_dependencies_map(dataset)

    # get list of all pos tags used in the dataset
    pos_map = pipeline.get_part_of_speech_map(dataset)

    # split it by sentence, potentially include negative samples
    sentences_dataset = pipeline.split_dataset_into_sentences(
        *dataset, include_negative_samples=opts.include_negative_samples
    )

    # split sentences between train and test according to the
    # official dataset split
    train_sentences, validation_sentences = pipeline.split_train_test_sentences(
        opts.subtask, sentences_dataset
    )

    test_dataset = pipeline.load_abstracts_relations(opts.subtask, load_test=True)
    test_sentences = pipeline.split_dataset_into_sentences(
        *test_dataset, include_negative_samples=opts.include_negative_samples
    )

    if opts.evaluate_output:
        evaluate_dataset = pipeline.load_abstracts_relations(
            opts.subtask, load_test=True)
        evaluate_sentences_dataset = pipeline.split_dataset_into_sentences(
            *evaluate_dataset,
            include_negative_samples=opts.include_negative_samples
        )
    else:
        # so that static code analyzers don't freak out!
        evaluate_sentences_dataset = None

    # get distribution info for entities in training set
    ent_distr = pipeline.get_distribution_ent_length(train_sentences)

    # get the mxnet context (aka cpu or gpu) as
    # provided by the user. if none is provided, use cpu0
    context = mxnet_utils.get_context_from_string(opts.mxnet_context)

    # path to embeddings file in word2vec text format
    # as specified by the user
    embeddings_path = os.path.expanduser(EMBEDDINGS_PATHS[opts.embeddings_type])


    # download embeddings from google drive
    embeddings_path = download_embeddings(path, opts.emb_type)


    # execute mxnet operations accoring in specified context
    with context:
        # load embeddings and vocabulary
        vocabulary, embeddings = \
            mxnet_utils.word2vec_mxnet_embedding_initializer(
                embeddings_path, max_embeddings=opts.max_embeddings
            )

        # get training data; has to be executed after vocabulary and
        # embeddings (which need to be placed on the GPU if specified,
        # hence the context) are loaded.
        train_data = prepare_data_for_net(
            vocabulary, train_sentences, labels_map,
            dependencies_map=dependencies_map, pos_map=pos_map,
            include_entities_nodes=opts.include_entities_nodes,
            include_entities_children=opts.include_entities_children,
            entity_length_distribution=ent_distr,
            case_sensitive=opts.case_sensitive
        )

        # doing the same thing, but with test data
        test_data = prepare_data_for_net(
            vocabulary, test_sentences, labels_map,
            dependencies_map=dependencies_map, pos_map=pos_map,
            include_entities_children=opts.include_entities_children,
            include_entities_nodes=opts.include_entities_nodes,
            entity_length_distribution=ent_distr,
            case_sensitive=opts.case_sensitive
        )

        # doing the same thing, but with test data
        validation_data = prepare_data_for_net(
            vocabulary, validation_sentences, labels_map,
            dependencies_map=dependencies_map, pos_map=pos_map,
            include_entities_children=opts.include_entities_children,
            include_entities_nodes=opts.include_entities_nodes,
            entity_length_distribution=ent_distr,
            case_sensitive=opts.case_sensitive
        )

        # get stats abt average size of parse tree
        parse_tree_lengths = [
            len(t) for _, _, t, *_ in
            itertools.chain(train_data, test_data)
        ]
        print('[info] parse tree length: {:.2f} +/- {:.2f}'.format(
            np.mean(parse_tree_lengths), np.std(parse_tree_lengths)
        ))

        max_tree_height = max(
            max(t[4]) for t in train_data
        ) + 1
        max_tree_height = (
            max_tree_height if 'height' in opts.extra_features else 0
        )

        dependencies_num = \
            len(dependencies_map) if 'dep' in opts.extra_features else 0

        pos_num = \
            len(pos_map) if 'pos' in opts.extra_features else 0

        include_ent_len = \
            True if 'ent-len' in opts.extra_features else False

        net = Net(
            embeddings,
            len(labels_map),
            dropout=opts.dropout,
            trainable_embeddings=opts.trainable_embeddings,
            dependencies_num=dependencies_num,
            part_of_speech_num=pos_num,
            include_ent_len=include_ent_len,
            max_tree_height=max_tree_height
        )
        net.initialize()

        # loos and trainer initialized here
        softmax_cross_entropy_labels = mx.gluon.loss.SoftmaxCrossEntropyLoss()

        trainer = mx.gluon.Trainer(
            net.collect_params(),
            'adam',
            {'learning_rate': opts.learning_rate}
        )

        # object to calculate F1 metric for the dataset
        f1_score_class = mxnet_utils.F1Score(num_classes=len(labels_map))

        for epoch in range(1, opts.epochs + 1):

            # random.shuffle(train_data)

            cumulative_loss = total_steps = 0
            probs, labels = [], []

            for sample in tqdm.tqdm(train_data, desc='Epoch {}'.format(epoch)):
                with mx.autograd.record():
                    (
                        tokens,             # the tokens in sentence
                        deps,               # dependency tags
                        pos,                # part of speech tags
                        ent_lens,           # length of the input entities
                        dist_from_tree,     # distance from root of subtree
                        tree,               # the subtree
                        entities,           # indication for entity location
                        label               # the label for this sample
                     )= sample

                    tokens = mx.nd.array(tokens)
                    entities = mx.nd.array(entities)
                    idx = mx.nd.array([tree.idx])
                    adj = mx.nd.array(tree.to_array())
                    deps = mx.nd.array(deps)
                    pos = mx.nd.array(pos)
                    ent_lens = mx.nd.array(ent_lens)
                    dist_from_tree = mx.nd.array(dist_from_tree)

                    out = net(
                        tokens, deps, pos, ent_lens, dist_from_tree,
                        adj, entities, idx, True
                    )

                    probs.append(out)
                    labels.append([label])

                if len(probs) == opts.batch_size:
                    total_steps += opts.batch_size

                    with mx.autograd.record():
                        probs = mx.nd.concat(*probs, dim=0)
                        labels = mx.nd.array(labels)
                        loss = softmax_cross_entropy_labels(probs, labels)

                        if opts.include_negative_samples:
                            factor = (mx.nd.argmax(probs, axis=1) == 0) * 9 + 1
                            loss = mx.nd.multiply(loss, factor)

                    loss.backward()
                    trainer.step(opts.batch_size)
                    cumulative_loss += mx.nd.sum(loss).asscalar()

                    pred_labels = mx.nd.argmax(probs, axis=1)
                    f1_score_class.update(preds=pred_labels, labels=labels)

                    probs, labels = [], []

            # get precision, recall, and F1 score for the two
            # subtasks on the training set for this epoch
            prec, recall, f1 = map(
                lambda arr: mx.nd.mean(arr).asscalar() * 100,
                f1_score_class.get()
            )

            # also calculate average loss
            avg_loss = cumulative_loss / total_steps

            # print everything
            msg = (
                'Epoch {e} // training data // avg_loss={l:.4f}\n'
                'Classification: P={p:.2f}  R={r:.2f}  F1={f:.2f}'
            ).format(
                e=epoch, l=avg_loss, p=prec, r=recall, f=f1
            )
            print(msg)

            if opts.validate_every > 0 and epoch % opts.validate_every == 0:

                if opts.error_analysis_path:
                    p = '{}{}.{}.txt'.format(
                        os.path.splitext(opts.error_analysis_path)[0],
                        'val', epoch
                    )
                else:
                    p = None

                evaluate_on_test_data(
                    net, validation_sentences, validation_data, labels_map,
                    output_for_error_analysis=p,
                )

            if opts.test_every > 0 and epoch % opts.test_every == 0:

                if opts.error_analysis_path:
                    p = '{}{}.{}.txt'.format(
                        os.path.splitext(opts.error_analysis_path)[0],
                        'test', epoch
                    )
                else:
                    p = None

                evaluate_on_test_data(
                    net, test_sentences, test_data, labels_map,
                    output_for_error_analysis=p,
                )

        if opts.evaluate_output:
            evaluate_data = prepare_data_for_net(
                vocabulary, evaluate_sentences_dataset, labels_map,
                dependencies_map=dependencies_map, pos_map=pos_map,
                include_entities_children=opts.include_entities_children,
                include_entities_nodes=opts.include_entities_nodes,
                entity_length_distribution=ent_distr,
                case_sensitive=opts.case_sensitive
            )

            evaluate_on_test_data(
                net, evaluate_sentences_dataset, evaluate_data, labels_map,
                evaluate_output=opts.evaluate_output
            )


if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument(
        '-s', '--subtask',
        default='1.1', choices=['1.1', '1.2', '2']
    )
    ap.add_argument(
        '-c', '--mxnet-context',
        default='cpu0'
    )
    ap.add_argument(
        '-C', '--include-entities-children',
        default=False, action='store_true'
    )
    ap.add_argument(
        '-e', '--embeddings-type',
        default='wikinews', choices=['arxiv', 'wiki', 'both']
    )
    ap.add_argument(
        '-r', '--learning-rate',
        default=0.002, type=float
    )
    ap.add_argument(
        '-p', '--epochs',
        default=100, type=int
    )

    ap.add_argument(
        '-d', '--dropout',
        default=0.2, type=float
    )
    ap.add_argument(
        '-E', '--trainable-embeddings',
        default=False, action='store_true'
    )
    ap.add_argument(
        '-n', '--include-negative-samples',
        action='store_true'
    )
    ap.add_argument(
        '-b', '--batch-size',
        default=16, type=int
    )
    ap.add_argument(
        '-l', '--labels-map',
        default=None, type=json_dict
    )
    ap.add_argument(
        '--epsilon',
        default=1e-12, type=float
    )
    ap.add_argument(
        '-V', '--validate-every',
        default=20, type=float
    )
    ap.add_argument(
        '-T', '--test-every',
        default=0, type=float
    )
    ap.add_argument(
        '-f', '--extra-features',
        nargs='+', choices=['pos', 'dep', 'ent-len', 'height'], default=[]
    )
    ap.add_argument(
        '--resnet',
        action='store_true', default=False
    )
    ap.add_argument(
        '--include-entities-nodes',
        action='store_true', default=False
    )

    ap.add_argument(
        '--error-analysis-path', default=None
    )
    ap.add_argument(
        '--case-sensitive', default=False, action='store_true'
    )
    ap.add_argument(
        '--max-embeddings', type=int, default=-1
    )

    ap.add_argument(
        '--evaluate-output', default=None
    )

    parsed_options = ap.parse_args()

    if parsed_options.resnet:
        raise NotImplementedError('--resnet not implemented')

    main(opts=parsed_options)
