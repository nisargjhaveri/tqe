from collections import Counter, defaultdict

import numpy as np
from sklearn.model_selection import ShuffleSplit

import logging
logger = logging.getLogger("common")


class WordIndexTransformer(object):
    def __init__(self, vocab_size=None):
        self.nextIndex = 1
        self.maxVocabSize = vocab_size
        self.vocabMap = {}
        self.wordCounts = Counter()

        self.finalized = False

    def _getIndex(self, token):
        return self.vocabMap.get(token, 0)

    def fit(self, sentences):
        if self.finalized:
            raise ValueError("Cannot fit after the transformer is finalized")

        for sentence in sentences:
            self.wordCounts.update(sentence)

        return self

    def finalize(self):
        if self.finalized:
            return self

        for token, count in self.wordCounts.most_common():
            if token in self.vocabMap:
                continue

            if (self.maxVocabSize is None
                    or self.nextIndex <= self.maxVocabSize):
                self.vocabMap[token] = self.nextIndex
                self.nextIndex += 1

        self.finalized = True
        self.wordCounts = Counter()

        return self

    def transform(self, sentences):
        self.finalize()

        transformedSentences = []
        for sentence in sentences:
            transformedSentences.append(
                np.array([self._getIndex(token) for token in sentence]))

        return np.array(transformedSentences)

    def extend(self, vocab_size=None):
        self.vocabSize = vocab_size
        self.finalized = False

        return self

    def vocab_size(self):
        return self.nextIndex

    def vocab_map(self):
        return self.vocabMap


def _preprocessSentences(sentences, lower=True, tokenize=True):
    def _processSentence(sentece):
        sentence = sentece.strip()

        if lower:
            sentence = sentence.lower()

        if tokenize:
            if callable(tokenize):
                sentence = np.array(tokenize(sentence), dtype=object)
            else:
                sentence = np.array(sentence.split(), dtype=object)

        return sentence

    return np.array(map(_processSentence, sentences), dtype=object)


def _loadSentences(filePath, lower=True, tokenize=True):
    with open(filePath) as lines:
        lines = map(lambda l: l.decode('utf-8'), list(lines))
        sentences = _preprocessSentences(lines,
                                         lower=lower, tokenize=tokenize)

    return sentences


def _loadData(fileBasename, devFileSuffix=None, testFileSuffix=None,
              lower=True, tokenize=True):
    targetPath = fileBasename + ".hter"
    srcSentencesPath = fileBasename + ".src"
    mtSentencesPath = fileBasename + ".mt"
    refSentencesPath = fileBasename + ".ref"

    srcSentences = _loadSentences(srcSentencesPath, lower, tokenize)
    mtSentences = _loadSentences(mtSentencesPath, lower, tokenize)
    refSentences = _loadSentences(refSentencesPath, lower, tokenize)

    y = np.clip(np.loadtxt(targetPath), 0, 1)

    if (testFileSuffix or devFileSuffix) and \
            not (testFileSuffix and devFileSuffix):
        raise ValueError("You have to specify both dev and test file suffix")

    if devFileSuffix and testFileSuffix:
        splitter = ShuffleSplit(n_splits=1, test_size=0, random_state=42)
        train_index, _ = splitter.split(srcSentences).next()

        srcSentencesDev = _loadSentences(srcSentencesPath + devFileSuffix,
                                         lower, tokenize)
        mtSentencesDev = _loadSentences(mtSentencesPath + devFileSuffix,
                                        lower, tokenize)
        refSentencesDev = _loadSentences(refSentencesPath + devFileSuffix,
                                         lower, tokenize)

        srcSentencesTest = _loadSentences(srcSentencesPath + testFileSuffix,
                                          lower, tokenize)
        mtSentencesTest = _loadSentences(mtSentencesPath + testFileSuffix,
                                         lower, tokenize)
        refSentencesTest = _loadSentences(refSentencesPath + testFileSuffix,
                                          lower, tokenize)

        y_dev = np.clip(np.loadtxt(targetPath + devFileSuffix), 0, 1)
        y_test = np.clip(np.loadtxt(targetPath + testFileSuffix), 0, 1)
    else:
        splitter = ShuffleSplit(n_splits=1, test_size=.2, random_state=42)
        train_index, dev_index = splitter.split(srcSentences).next()

        dev_len = len(dev_index) / 2

        srcSentencesDev = srcSentences[dev_index[:dev_len]]
        mtSentencesDev = mtSentences[dev_index[:dev_len]]
        refSentencesDev = refSentences[dev_index[:dev_len]]

        srcSentencesTest = srcSentences[dev_index[dev_len:]]
        mtSentencesTest = mtSentences[dev_index[dev_len:]]
        refSentencesTest = refSentences[dev_index[dev_len:]]

        y_dev = y[dev_index[:dev_len]]
        y_test = y[dev_index[dev_len:]]

    srcSentencesTrain = srcSentences[train_index]
    mtSentencesTrain = mtSentences[train_index]
    refSentencesTrain = refSentences[train_index]

    y_train = y[train_index]

    X_train = {
        "src": srcSentencesTrain,
        "mt": mtSentencesTrain,
        "ref": refSentencesTrain
    }
    X_dev = {
        "src": srcSentencesDev,
        "mt": mtSentencesDev,
        "ref": refSentencesDev
    }
    X_test = {
        "src": srcSentencesTest,
        "mt": mtSentencesTest,
        "ref": refSentencesTest
    }

    return X_train, y_train, X_dev, y_dev, X_test, y_test


def pad_sequences(sequences, maxlen=None, num_buckets=1, lengths=None,
                  **kwargs):
    from keras.preprocessing.sequence import pad_sequences
    if num_buckets <= 1 and not lengths:
        return pad_sequences(sequences, maxlen, **kwargs)
    else:
        if not lengths:
            if not maxlen:
                maxlen = max(map(len, sequences))

            bucket_size = maxlen / num_buckets

            def get_padlen(s):
                return min(
                    int(np.ceil(len(s) / float(bucket_size))) * bucket_size,
                    maxlen
                )

            lengths = map(get_padlen, sequences)

        return np.array(
                map(lambda s, l: pad_sequences([s], l, **kwargs)[0],
                    sequences, lengths)
                )


def getBatchGenerator(*args, **kwargs):
    """
    X is assumed to be list of inputs
    y is assumed to be list of outputs
    """
    from keras.utils import Sequence

    class BatchGeneratorSequence(Sequence):
        def __init__(self, X, y=None, key=lambda x: x, batch_size=None):
            self.batch_size = batch_size
            self.X = X
            self.y = y

            groupingKyes = map(key, zip(*X))

            groups = {}
            for i, key in enumerate(groupingKyes):
                groups.setdefault(key, []).append(i)

            self.batches = []
            for group in groups.values():
                num_samples = len(group)
                batch_size = num_samples if not batch_size else batch_size

                batches = [
                    group[i:i + batch_size]
                    for i in xrange(0, num_samples, batch_size)
                ]

                self.batches.extend(batches)

        def __len__(self):
            # print len(self.batches)
            return len(self.batches)

        def __getitem__(self, idx):
            batch_idx = self.batches[idx]

            batch = [np.array(x_i[batch_idx].tolist()) for x_i in self.X]

            if self.y:
                batch = (
                    batch,
                    [np.array(y_i[batch_idx].tolist()) for y_i in self.y]
                )

            return batch

        def align(self, y):
            alignment = sum(self.batches, [])
            return y[alignment]

        def alignOriginal(self, y):
            alignment = sum(self.batches, [])
            y_aligned = y[:]
            for i, y_curr in zip(alignment, y):
                y_aligned[i] = y_curr

            return y_aligned

    return BatchGeneratorSequence(*args, **kwargs)


def _prepareInput(workspaceDir, dataName,
                  srcVocabTransformer, refVocabTransformer,
                  max_len, num_buckets,
                  train_vocab=True,
                  lower=True, tokenize=True,
                  devFileSuffix=None, testFileSuffix=None,
                  ):
    import os

    logger.info("Loading data")

    X_train, y_train, X_dev, y_dev, X_test, y_test = _loadData(
                    os.path.join(workspaceDir, "tqe." + dataName),
                    devFileSuffix, testFileSuffix,
                    lower=lower, tokenize=tokenize,
                )

    logger.info("Transforming sentences to onehot")

    if train_vocab:
        srcVocabTransformer \
            .fit(X_train['src']) \
            .fit(X_dev['src']) \
            .fit(X_test['src'])

    srcSentencesTrain = srcVocabTransformer.transform(X_train['src'])
    srcSentencesDev = srcVocabTransformer.transform(X_dev['src'])
    srcSentencesTest = srcVocabTransformer.transform(X_test['src'])

    if train_vocab:
        refVocabTransformer.fit(X_train['mt']) \
                           .fit(X_dev['mt']) \
                           .fit(X_test['mt']) \
                           .fit(X_train['ref']) \
                           .fit(X_dev['ref']) \
                           .fit(X_test['ref'])

    mtSentencesTrain = refVocabTransformer.transform(X_train['mt'])
    mtSentencesDev = refVocabTransformer.transform(X_dev['mt'])
    mtSentencesTest = refVocabTransformer.transform(X_test['mt'])
    refSentencesTrain = refVocabTransformer.transform(X_train['ref'])
    refSentencesDev = refVocabTransformer.transform(X_dev['ref'])
    refSentencesTest = refVocabTransformer.transform(X_test['ref'])

    def getMaxLen(listOfsequences):
        return max([max(map(len, sequences)) for sequences in listOfsequences
                    if len(sequences)])

    srcMaxLen = min(getMaxLen([srcSentencesTrain, srcSentencesDev]), max_len)
    refMaxLen = min(getMaxLen([mtSentencesTrain, mtSentencesDev,
                               refSentencesTrain, refSentencesDev]), max_len)

    pad_args = {'num_buckets': num_buckets}
    X_train = {
        "src": pad_sequences(srcSentencesTrain, maxlen=srcMaxLen, **pad_args),
        "mt": pad_sequences(mtSentencesTrain, maxlen=refMaxLen, **pad_args),
        "ref": pad_sequences(refSentencesTrain, maxlen=refMaxLen, **pad_args)
    }

    X_dev = {
        "src": pad_sequences(srcSentencesDev, maxlen=srcMaxLen, **pad_args),
        "mt": pad_sequences(mtSentencesDev, maxlen=refMaxLen, **pad_args),
        "ref": pad_sequences(refSentencesDev, maxlen=refMaxLen, **pad_args)
    }

    X_test = {
        "src": pad_sequences(srcSentencesTest, maxlen=srcMaxLen, **pad_args),
        "mt": pad_sequences(mtSentencesTest, maxlen=refMaxLen, **pad_args),
        "ref": pad_sequences(refSentencesTest, maxlen=refMaxLen, **pad_args)
    }

    return X_train, y_train, X_dev, y_dev, X_test, y_test


def _extendVocabFor(workspaceDir, dataName,
                    srcVocabTransformer, refVocabTransformer,
                    lower=True, tokenize=True,
                    devFileSuffix=None, testFileSuffix=None,
                    ):
    import os

    logger.info("Loading pretrain_for data")

    X_train, y_train, X_dev, y_dev, X_test, y_test = _loadData(
                    os.path.join(workspaceDir, "tqe." + dataName),
                    devFileSuffix, testFileSuffix,
                    lower=lower, tokenize=tokenize,
                )

    logger.info("Extending vocab")

    srcVocabTransformer \
        .extend() \
        .fit(X_train['src']) \
        .fit(X_dev['src']) \
        .fit(X_test['src']) \
        .finalize()

    refVocabTransformer \
        .extend() \
        .fit(X_train['mt']) \
        .fit(X_dev['mt']) \
        .fit(X_test['mt']) \
        .fit(X_train['ref']) \
        .fit(X_dev['ref']) \
        .fit(X_test['ref']) \
        .finalize()


def getBinaryThreshold(binary_threshold, y_train):
    if binary_threshold is None:
        binary_threshold = np.mean(y_train)

    return binary_threshold


def binarize(threshold, *args):
    return map(
                lambda x: np.where(x >= threshold, 1, 0),
                args
            )


def _get_embedding_path(workspaceDir, model):
    import os
    return os.path.join(workspaceDir,
                        "fastText",
                        ".".join([model, "bin"])
                        ) if model else None


fastTextCache = defaultdict(dict)


def get_fastText_embeddings(fastText_file, vocabTransformer, embedding_size):
    ft_cache = fastTextCache[fastText_file]
    embedding_matrix = np.zeros(
                        shape=(vocabTransformer.vocab_size(), embedding_size)
                    )

    missingTokens = []
    for token, i in vocabTransformer.vocab_map().items():
        if token in ft_cache:
            embedding_matrix[i] = ft_cache[token]
        else:
            missingTokens.append((token, i))

    if len(missingTokens):
        import fastText
        ft_model = fastText.load_model(fastText_file)

        for token, i in missingTokens:
            ft_cache[token] = ft_model.get_word_vector(token)
            embedding_matrix[i] = ft_cache[token]

    return embedding_matrix


def TimeDistributedSequential(layers, inputs, name=None):
    from keras.layers import TimeDistributed

    layer_names = ["_".join(["td", layer.name]) for layer in layers]

    if name:
        layer_names[-1] = name

    input = inputs
    for layer, layer_name in zip(layers, layer_names):
        input = TimeDistributed(
                    layer, name=layer_name
                )(input)

    return input


def _printModelSummary(logger, model, name, plot=False):
    if plot:
        from keras.utils import plot_model
        plot_model(model, to_file=(name if name else "model") + ".png")

    model_summary = ["Printing model summary"]

    if name:
        model_summary += ["Model " + name]

    def summary_capture(line):
        model_summary.append(line)

    model.summary(print_fn=summary_capture)
    logger.info("\n".join(model_summary))


def pearsonr(y_true, y_pred):
    # From https://github.com/scipy/scipy/blob/v0.14.0/scipy/stats/stats.py
    #
    # x = np.asarray(x)
    # y = np.asarray(y)
    # n = len(x)
    # mx = x.mean()
    # my = y.mean()
    # xm, ym = x-mx, y-my
    # r_num = np.add.reduce(xm * ym)
    # r_den = np.sqrt(ss(xm) * ss(ym))
    # r = r_num / r_den
    #
    # # Presumably, if abs(r) > 1, then it is only some small artifact of
    # # floating point arithmetic.
    # r = max(min(r, 1.0), -1.0)
    # df = n-2
    # if abs(r) == 1.0:
    #     prob = 0.0
    # else:
    #     t_squared = r*r * (df / ((1.0 - r) * (1.0 + r)))
    #     prob = betai(0.5*df, 0.5, df / (df + t_squared))
    # return r, prob

    import keras.backend as K

    x = y_true
    y = y_pred
    # n = x.shape[0]
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    r_den = K.sqrt(K.sum(xm * xm) * K.sum(ym * ym))
    r = K.switch(K.not_equal(r_den, 0), r_num / r_den, 0)

    # Presumably, if abs(r) > 1, then it is only some small artifact of
    # floating point arithmetic.
    r = K.clip(r, -1.0, 1.0)
    # df = n-2
    # if abs(r) == 1.0:
    #     prob = 0.0
    # else:
    #     t_squared = r*r * (df / ((1.0 - r) * (1.0 + r)))
    #     prob = betai(0.5*df, 0.5, df / (df + t_squared))
    return r


def getStatefulPearsonr(**kwargs):
    from keras.layers import Layer
    import keras.backend as K

    class StatefulPearsonr(Layer):
        def __init__(self, **kwargs):
            super(StatefulPearsonr, self).__init__(name="pearsonr", **kwargs)

            self.stateful = True

            self.n = K.variable(value=0, dtype='int')
            self.sum_xy = K.variable(value=0, dtype='float')
            self.sum_x = K.variable(value=0, dtype='float')
            self.sum_y = K.variable(value=0, dtype='float')
            self.sum_x_2 = K.variable(value=0, dtype='float')
            self.sum_y_2 = K.variable(value=0, dtype='float')

        def reset_states(self):
            K.set_value(self.n, 0)
            K.set_value(self.sum_xy, 0)
            K.set_value(self.sum_x, 0)
            K.set_value(self.sum_y, 0)
            K.set_value(self.sum_x_2, 0)
            K.set_value(self.sum_y_2, 0)

        def __call__(self, y_true, y_pred):
            x = y_true
            y = y_pred

            n = self.n + K.shape(x)[0]
            sum_xy = self.sum_xy + K.sum(x * y)
            sum_x = self.sum_x + K.sum(x)
            sum_y = self.sum_y + K.sum(y)
            sum_x_2 = self.sum_x_2 + K.sum(x * x)
            sum_y_2 = self.sum_y_2 + K.sum(y * y)

            self.add_update(K.update_add(self.n, K.shape(x)[0]),
                            inputs=[y_true, y_pred])
            self.add_update(K.update_add(self.sum_xy, K.sum(x * y)),
                            inputs=[y_true, y_pred])
            self.add_update(K.update_add(self.sum_x, K.sum(x)),
                            inputs=[y_true, y_pred])
            self.add_update(K.update_add(self.sum_y, K.sum(y)),
                            inputs=[y_true, y_pred])
            self.add_update(K.update_add(self.sum_x_2, K.sum(x * x)),
                            inputs=[y_true, y_pred])
            self.add_update(K.update_add(self.sum_y_2, K.sum(y * y)),
                            inputs=[y_true, y_pred])

            r_num = (n * sum_xy) - (sum_x * sum_y)
            r_den = (K.sqrt((n * sum_x_2) - (sum_x * sum_x))
                     * K.sqrt((n * sum_y_2) - (sum_y * sum_y)))
            r = r_num / r_den

            # Presumably, if abs(r) > 1, then it is only some small artifact of
            # floating point arithmetic.
            r = K.clip(r, -1.0, 1.0)
            return r

    return StatefulPearsonr(**kwargs)


def getStatefulAccuracy(**kwargs):
    from keras.layers import Layer
    import keras.backend as K

    class StatefulAccuracy(Layer):
        def __init__(self, **kwargs):
            super(StatefulAccuracy, self).__init__(name="acc", **kwargs)

            self.stateful = True

            self.correct = K.variable(value=0, dtype='float')
            self.total = K.variable(value=0, dtype='float')

        def reset_states(self):
            K.set_value(self.correct, 0)
            K.set_value(self.total, 0)

        def __call__(self, y_true, y_pred):
            total = self.total + K.shape(y_true)[0]
            correct = self.correct + K.sum(K.equal(y_true, K.round(y_pred)))

            self.add_update(K.update_add(self.total,
                            K.shape(y_true)[0]),
                            inputs=[y_true, y_pred])
            self.add_update(K.update_add(self.correct,
                            K.sum(K.equal(y_true, K.round(y_pred)))),
                            inputs=[y_true, y_pred])

            return correct / total

    return StatefulAccuracy(**kwargs)


def _binaryClassificationEvaluate(y_pred, y_true, output=True):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.metrics import precision_recall_fscore_support

    y_pred_bin, = binarize(0.5, y_pred)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    precision, recall, f1, support = \
        precision_recall_fscore_support(y_true, y_pred_bin)

    predicted = np.sum(y_pred_bin)

    if output:
        print "\t".join([
            "MSE", "MAE", "Prec.", "Recall", "F1", "True", "Predicted",
        ])
        print "\t".join([
            ("%1.5f" % mse),
            ("%1.5f" % mae),
            ("%1.5f" % precision[1]),
            ("%1.5f" % recall[1]),
            ("%1.5f" % f1[1]),
            ("%d" % support[1]),
            ("%d" % predicted),
        ])

    return {
        "MSE": mse,
        "MAE": mae,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _regressionEvaluate(y_pred, y_true, output=True):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from scipy.stats import pearsonr
    from scipy.stats import spearmanr

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    pearsonR = pearsonr(y_pred, y_true)
    spearmanR = spearmanr(y_pred, y_true)

    if output:
        print "\t".join([
            "MSE", "MAE", "PCC", "p-value  ", "SCC", "p-value  "
        ])
        print "\t".join([
            ("%1.5f" % mse),
            ("%1.5f" % mae),
            ("%1.5f" % pearsonR[0]),
            ("%.3e" % pearsonR[1]),
            ("%1.5f" % spearmanR[0]),
            ("%.3e" % spearmanR[1]),
        ])

    return {
        "MSE": mse,
        "MAE": mae,
        "pearsonR": pearsonR,
        "spearmanR": spearmanR
    }


def evaluate(y_pred, y_true, binary=False, **kwargs):
    if binary:
        return _binaryClassificationEvaluate(y_pred, y_true, **kwargs)
    else:
        return _regressionEvaluate(y_pred, y_true, **kwargs)
