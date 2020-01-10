# built-in modules
import re
import typing

# installed modules
import tqdm
import mxnet as mx
import mxnet.autograd


def get_context_from_string(ctx_str: str) -> mx.context.Context:
    """
    Parse a mxnet Context object from string, e.g. 'gpu0'
    to `mx.Context(mx.gpu(0))`.

    :param ctx_str: The string to parse
    :type ctx_str: str

    :return: a mxnet Context object
    :rtype: mx.context.Context
    """
    device = re.sub(r'[^a-z]', '', ctx_str.lower())

    try:
        device_id = int(re.search(r'\d+', ctx_str).group())
    except AttributeError:
        device_id = 0

    if device == 'cpu':
        return mx.Context(mx.cpu(device_id))
    elif device == 'gpu':
        return mx.Context(mx.gpu(device_id))

    raise TypeError('Context {} not recognized'.format(ctx_str))


def word2vec_mxnet_embedding_initializer(
        path, context: typing.Union[None, mx.context.Context]=None,
        max_embeddings=-1
) -> typing.Tuple[dict, mx.ndarray.ndarray.NDArray]:
    """
    Initializes word embedding into mxnet-compatible data matrix.

    :param path: path to word embedding file in word2vec text format
    :type path: str

    :param context: context for mxnet vector initialization. Can be None
        or mx.context.Context. If None, current context is used (cpu0 if
        context is not specified).
    :type context: Union[None, mx.context.Context]

    :return: tuple containing the following items
        -   embedding_vocab: vocabulary to look up index of terms in
            in word embedding matrix
        -   embedding_mat: mxnet NDArray containing embedding values
    :rtype: Tuple[dict, mx.ndarray.ndarray.NDArray]
    """

    embedding_vocab = {'<UNK>': 0}
    context = mx.context.current_context() if context is None else context
    with open(path) as f, context:
        vocab_size, emb_size = map(int, f.readline().strip().split())
        vocab_size += 1
        embedding_mat = mx.nd.zeros((vocab_size, emb_size), dtype='float32')
        embedding_mat[0] = mx.nd.random_normal(shape=(emb_size,)) / 10e3

        iterator = tqdm.tqdm(
            enumerate(f, start=1),
            total=vocab_size,
            desc='loading embeddings',
            initial=1
        )
        for i, ln in iterator:

            word, *raw_vec = ln.rstrip().split(' ')
            embedding_mat[i] = mx.nd.array(raw_vec, dtype='float32')
            embedding_vocab[word] = i

            if max_embeddings > 0 and i > max_embeddings:
                break

    return embedding_vocab, embedding_mat



class F1Score:
    """
    Object to keep track and calculate precision, recall and F1 scores.
    """
    def __init__(self, num_classes: int, eps: float=1e-12):
        """
        Initialize F1 score object

        :param num_classes: number of total classes to consider
            for computing F1 score.
        :type num_classes: int

        :param eps: value to use in order to prevent divide by
            zero errors. By default, 1e-12 is used. You shouldn't
            need to modify this
        :type eps: float
        """
        self.all_labels = mx.nd.arange(num_classes)
        self.confusion_matrix = mx.nd.zeros((num_classes, num_classes))
        self.eps = eps

    def update(self, preds: mx.nd.NDArray, labels: mx.nd.NDArray):
        """
        Update confusion table used to compute scores.

        :param preds: predicted classes for samples in batch.
        :type preds: mx.ndarray.NDArray

        :param labels: actual labels for samples in batch.
        :type labels: mx.ndarray.NDArray
        """
        preds = preds.reshape((-1, 1))
        labels = labels.reshape((-1, 1))

        preds_binary = (preds == self.all_labels)
        labels_binary = (labels == self.all_labels)

        self.confusion_matrix += mx.nd.dot(labels_binary.T, preds_binary)

    def get(self) -> typing.Union[mx.nd.NDArray, mx.nd.NDArray, mx.nd.NDArray]:
        """
        Returns precision, recall, and F1 scores for each class.

        :return: Tuple containing mxnet arrays with precision, recall,
            and F1 score for each class.
        :rtype: typing.Union[mx.nd.NDArray, mx.nd.NDArray, mx.nd.NDArray]
        """
        # get the true positive for each class (that is, all
        # elements on the diagonal)
        tp = self.confusion_matrix[self.all_labels, self.all_labels]

        # count the number of true elements per class
        # (i.e., true positive + false negative)
        tp_fn = mx.nd.sum(self.confusion_matrix, axis=1)

        # count the number of elements per class that have
        # been labeled as positive (i.e., true positive +
        # false positive)
        tp_fp = mx.nd.sum(self.confusion_matrix, axis=0)

        # calculate precision, recall, and F1 score
        recall = tp / (tp_fn + self.eps)
        precision = tp / (tp_fp + self.eps)
        f1_score = 2 * precision * recall / (precision + recall + self.eps)

        return precision, recall, f1_score



def apply_dropout(mat, drop_probability):
    keep_probability = 1 - drop_probability
    mask = (
            mx.nd.random_uniform(0, 1.0, mat.shape, ctx=mat.context) <
            keep_probability
    )
    #############################
    #  Avoid division by 0 when scaling
    #############################
    if keep_probability > 0.0:
        scale = (1 / keep_probability)
    else:
        scale = 0.0
    return mask * mat * scale


class TreeLSTM(mx.gluon.HybridBlock):
    """
    Child sum LSTM cell. Described in Tai, Kai Sheng, Richard Socher,
    and Christopher D. Manning. "Improved semantic representations from
    tree-structured long short-term memory networks." arXiv preprint
    arXiv:1503.00075 (2015).
    https://nlp.stanford.edu/pubs/tai-socher-manning-acl2015.pdf
    """
    def __init__(self, input_size, hidden_size, weight_init=None,
                 bias_init='zeros', dropout=0.0, **kwargs):
        super(TreeLSTM, self).__init__(**kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout

        # input gate layer
        self.Wi = self.params.get(
            name='Wi',
            shape=(hidden_size, input_size + hidden_size),
            init=weight_init
        )
        self.bi = self.params.get(
            name='bi',
            shape=(hidden_size,),
            init=bias_init
        )

        # forget gate layer
        self.Wf = self.params.get(
            name='Wf',
            shape=(hidden_size, hidden_size),
            init=weight_init
        )
        self.bf = self.params.get(
            name='bf',
            shape=(hidden_size,),
            init=bias_init
        )

        # update gate layer
        self.Wu = self.params.get(
            name='Wu',
            shape=(hidden_size, input_size + hidden_size),
            init=weight_init
        )
        self.bu = self.params.get(
            name='bu',
            shape=(hidden_size,),
            init=bias_init
        )

        # update gate layer
        self.Wo = self.params.get(
            name='Wo',
            shape=(hidden_size, input_size + hidden_size),
            init=weight_init
        )
        self.bo = self.params.get(
            name='bo',
            shape=(hidden_size,),
            init=bias_init
        )

    def drop(self, mat):
        if mx.autograd.is_training() and self.dropout > 0:
            mat = apply_dropout(mat, self.dropout)
        return mat

    def node_forward(
            self, input_values, children_outputs, children_states, params
    ):
        # sum of hidden states from children
        hj = sum(children_outputs)

        # concatenate the sum with current input
        xj_hj = mx.nd.concat(input_values, hj, dim=0)

        # input gate output
        ij = self.drop(
            mx.nd.sigmoid(mx.nd.dot(params['Wi'], xj_hj) + params['bi'])
        )

        # update gate output
        uj = self.drop(
            mx.nd.sigmoid(mx.nd.dot(params['Wu'], xj_hj) + params['bu'])
        )

        # create new cell state modifier
        ij_uj = mx.nd.multiply(ij, uj)

        # forget gates output
        fj = sum(
            mx.nd.multiply(
                self.drop(mx.nd.sigmoid(
                    mx.nd.dot(params['Wf'], hk) +
                    params['bf']
                )), ck
            ) for hk, ck in zip(children_states, children_outputs)
        )

        # new cell state
        cj = fj + ij_uj

        # new output modifier
        oj = self.drop(
            mx.nd.sigmoid(mx.nd.dot(params['Wo'], xj_hj) + params['bo'])
        )

        # new output state
        hj = mx.nd.multiply(mx.nd.tanh(cj), oj)

        return hj, cj

    def hybrid_forward(self, F, inputs, adj_matrix, idx, **params):
        node_children = adj_matrix[idx].reshape((-1,))

        if mx.nd.sum(node_children) > 0:
            children_outputs, children_states = zip(*(
                self.forward(inputs, adj_matrix, mx.nd.array([child]))
                for child, adj_value in enumerate(node_children)
                if adj_value > 0
            ))
        else:
            children_outputs = [mx.nd.zeros(shape=(self.hidden_size,))]
            children_states = [mx.nd.zeros(shape=(self.hidden_size,))]

        input_values = inputs[idx].reshape((-1,))
        return self.node_forward(
            input_values, children_outputs, children_states, params
        )


class EmbeddingInit(mx.init.Initializer):
    """
    Custom initializer for embeddings layer
    """
    def __init__(self, weights: mx.nd.NDArray):
        """
        Initialize initializer for embedding weights.

        :param weights: weights to initialize embeddigns with
        :type weights: mx.nd.NDArray
        """
        super(EmbeddingInit, self).__init__(weights=weights)
        self.weights = weights

    def _init_weight(self, name, arr):
        arr[:] = self.weights


def get_embedding_layer(
        vocab_size: int=None,
        embedding_size: int=None,
        embedding_weights: mx.nd.NDArray=None,
        weight_initializer: mx.initializer.Initializer=None,
        is_trainable=True,
        context: mx.context.Context=None
) -> mx.gluon.nn.Embedding:
    """
    Returns an embedding layer.

    :param vocab_size: size of the vocabulary. Not necessary if
        embedding_weights is provided.
    :type vocab_size: int

    :param embedding_size: size of the embeddings. Not necessary
        if embedding_weights is provided.
    :type embedding_size: int

    :param embedding_weights: weights to initialize embeddings
        layer. If not provided, embedding layer is initialized
        using weight_initializer or uniform initialization.

    :param weight_initializer: default initializer if
        embedding_weights are not provided.
    :type weight_initializer: mx.initializer.Initializer

    :param is_trainable: if True, the embeddings are set as trainable;
        if False, embeddings are kept constant during training
    :type is_trainable: bool

    :param context: mxnet context to use to create layers.
        if not provided, the current context is used
    :type context: mx.context.Context

    :return: embedding layer
    :rtype: mx.gluon.nn.Embedding
    """

    context = mx.context.current_context() if context is None else context

    not_enough_info = (
            embedding_weights is None and
            (vocab_size is None or embedding_size is None)
    )

    if not_enough_info:
        err_msg = (
            'Error: not enough parameters. '
            'Specify either embedding_weights '
            'or vocab_size and embeddings_size.'
        )
        raise ValueError(err_msg)

    if weight_initializer is None:
        weight_initializer = mx.init.Uniform

    with context:
        if embedding_weights is not None:
            vocab_size, embedding_size = embedding_weights.shape
            init = EmbeddingInit(embedding_weights)
        else:
            init = weight_initializer

        embedding_layer = mx.gluon.nn.Embedding(
            vocab_size, embedding_size, weight_initializer=init
        )
        if not is_trainable:
            embedding_layer.weight.grad_req = 'null'

    return embedding_layer
