import os
import shelve

from keras.layers import dot, average
from keras.layers import Input, Embedding, Dense, Activation, Lambda
from keras.layers import GRU, GRUCell, Bidirectional, RNN
from keras.models import Model
from keras.callbacks import EarlyStopping

import keras.backend as K

from keras.utils.generic_utils import CustomObjectScope

from .common import evaluate

from .common import _prepareInput, _extendVocabFor
from .common import WordIndexTransformer, _preprocessSentences
from .common import _printModelSummary, TimeDistributedSequential
from .common import pad_sequences, getBatchGenerator
from .common import getStatefulPearsonr, getStatefulAccuracy
from .common import get_fastText_embeddings, _get_embedding_path
from .common import getBinaryThreshold, binarize


import logging
logger = logging.getLogger("rnn")


class AttentionGRUCell(GRUCell):
    def __init__(self, units, *args, **kwargs):
        super(AttentionGRUCell, self).__init__(units, *args, **kwargs)

    def build(self, input_shape):
        self.constants_shape = None
        if isinstance(input_shape, list):
            if len(input_shape) > 1:
                self.constants_shape = input_shape[1:]
            input_shape = input_shape[0]

        cell_input_shape = list(input_shape)
        cell_input_shape[-1] += self.constants_shape[0][-1]
        cell_input_shape = tuple(cell_input_shape)

        super(AttentionGRUCell, self).build(cell_input_shape)

    def attend(self, query, attention_states):
        # Multiply query with each state per batch
        attention = K.batch_dot(
                        attention_states, query,
                        axes=(attention_states.ndim - 1, query.ndim - 1)
                    )

        # Take softmax to get weight per timestamp
        attention = K.softmax(attention)

        # Take weigthed average of attention_states
        context = K.batch_dot(attention, attention_states)

        return context

    def call(self, inputs, states, training=None, constants=None):
        context = self.attend(states[0], constants[0])

        inputs = K.concatenate([context, inputs])

        cell_out, cell_state = super(AttentionGRUCell, self).call(
                                            inputs, states, training=training)

        return cell_out, cell_state


def getModel(srcVocabTransformer, refVocabTransformer,
             embedding_size, gru_size,
             src_fastText, ref_fastText, train_embeddings,
             attention, summary_attention, use_estimator,
             model_inputs=None, verbose=False,
             ):
    src_vocab_size = srcVocabTransformer.vocab_size()
    ref_vocab_size = refVocabTransformer.vocab_size()

    src_embedding_kwargs = {}
    ref_embedding_kwargs = {}

    if src_fastText:
        logger.info("Loading fastText embeddings for source language")
        src_embedding_kwargs['weights'] = [get_fastText_embeddings(
                                src_fastText,
                                srcVocabTransformer,
                                embedding_size
                                )]

    if ref_fastText:
        logger.info("Loading fastText embeddings for target language")
        ref_embedding_kwargs['weights'] = [get_fastText_embeddings(
                                ref_fastText,
                                refVocabTransformer,
                                embedding_size
                                )]

    if verbose:
        logger.info("Creating model")

    if model_inputs:
        src_input, ref_input = model_inputs
    else:
        src_input = Input(shape=(None, ))
        ref_input = Input(shape=(None, ))

    src_embedding = Embedding(
                        output_dim=embedding_size,
                        input_dim=src_vocab_size,
                        mask_zero=True,
                        name="src_embedding",
                        trainable=train_embeddings,
                        **src_embedding_kwargs)(src_input)

    ref_embedding = Embedding(
                        output_dim=embedding_size,
                        input_dim=ref_vocab_size,
                        mask_zero=True,
                        name="ref_embedding",
                        trainable=train_embeddings,
                        **ref_embedding_kwargs)(ref_input)

    encoder = Bidirectional(
                    GRU(gru_size, return_sequences=True, return_state=True),
                    name="encoder"
            )(src_embedding)

    return_sequence = (use_estimator or summary_attention)
    if attention:
        attention_states = TimeDistributedSequential(
                                [Dense(gru_size, name="attention_state")],
                                encoder[0]
                            )

        with CustomObjectScope({'AttentionGRUCell': AttentionGRUCell}):
            decoder = Bidirectional(
                        RNN(AttentionGRUCell(gru_size),
                            return_sequences=return_sequence,
                            return_state=return_sequence),
                        name="decoder"
                    )(
                      ref_embedding,
                      constants=attention_states,
                      initial_state=encoder[1:]
                    )
    else:
        decoder = Bidirectional(
                    GRU(gru_size,
                        return_sequences=return_sequence,
                        return_state=return_sequence),
                    name="decoder"
                )(
                  ref_embedding,
                  initial_state=encoder[1:]
                )

    if use_estimator:
        decoder = Bidirectional(
                    GRU(gru_size,
                        return_sequences=summary_attention,
                        return_state=summary_attention),
                    name="estimator"
                )(decoder[0])

    if summary_attention:
        attention_weights = TimeDistributedSequential([
            Dense(gru_size, activation="tanh"),
            Dense(1, name="attention_weights"),
        ], decoder[0])

        # attention_weights = Reshape((-1,))(attention_weights)
        attention_weights = Lambda(
                    lambda x: K.reshape(x, (x.shape[0], -1,)),
                    output_shape=lambda input_shape: input_shape[:-1],
                    mask=lambda inputs, mask: mask,
                    name="reshape"
                    )(attention_weights)

        attention_weights = Activation(
                                "softmax",
                                name="attention_softmax"
                            )(attention_weights)

        quality_summary = dot([attention_weights, decoder[0]],
                              axes=(1, 1),
                              name="summary"
                              )
    else:
        quality_summary = decoder

    quality = Dense(1, name="quality")(quality_summary)

    model = Model(inputs=[src_input, ref_input],
                  outputs=[quality])

    if verbose:
        _printModelSummary(logger, model, "model")

    return model


def _getEnsembledModel(ensemble_count, **kwargs):
    if ensemble_count == 1:
        return getModel(verbose=True, **kwargs)

    src_input = Input(shape=(None, ))
    ref_input = Input(shape=(None, ))

    model_inputs = [src_input, ref_input]

    logger.info("Creating models to ensemble")
    models = [getModel(model_inputs=model_inputs, **kwargs)
              for _ in range(ensemble_count)]
    _printModelSummary(logger, models[0], "base_model")

    output = average([model([src_input, ref_input]) for model in models],
                     name='quality')

    model = Model(inputs=[src_input, ref_input],
                  outputs=output)

    _printModelSummary(logger, model, "ensembled_model")

    return model


def getEnsembledModel(binary, **kwargs):
    model = _getEnsembledModel(**kwargs)

    logger.info("Compiling model")
    if binary:
        loss = "binary_crossentropy"
        metrics = ["mae", getStatefulAccuracy()]
    else:
        loss = "mse"
        metrics = ["mse", "mae", getStatefulPearsonr()]

    model.compile(
            optimizer="adadelta",
            loss={
                "quality": loss
            },
            metrics={
                "quality": metrics
            }
        )

    return model


def train_model(workspaceDir, modelName, devFileSuffix, testFileSuffix,
                pretrain_for,
                pretrain_devFileSuffix, pretrain_testFileSuffix,
                pretrained_model,
                saveModel,
                binary, binary_threshold,
                batchSize, epochs, max_len, num_buckets, vocab_size,
                early_stop,
                **kwargs):
    logger.info("initializing TQE training")

    if pretrained_model:
        shelf = shelve.open(os.path.join(workspaceDir,
                                         "model." + pretrained_model), 'r')

        # Load vocab
        srcVocabTransformer = shelf['params']['srcVocabTransformer']
        refVocabTransformer = shelf['params']['refVocabTransformer']
        train_vocab = False

        # Don't load from fastText again
        kwargs['src_fastText'] = None
        kwargs['ref_fastText'] = None

        model_weights = shelf['weights']

        shelf.close()
    else:
        # Setup vocab
        srcVocabTransformer = WordIndexTransformer(vocab_size=vocab_size)
        refVocabTransformer = WordIndexTransformer(vocab_size=vocab_size)
        train_vocab = True

        # Set paths for fastText models
        kwargs['src_fastText'] = _get_embedding_path(workspaceDir,
                                                     kwargs['src_fastText'])
        kwargs['ref_fastText'] = _get_embedding_path(workspaceDir,
                                                     kwargs['ref_fastText'])

    X_train, y_train, X_dev, y_dev, X_test, y_test = _prepareInput(
                                        workspaceDir,
                                        modelName,
                                        srcVocabTransformer,
                                        refVocabTransformer,
                                        train_vocab=train_vocab,
                                        max_len=max_len,
                                        num_buckets=num_buckets,
                                        devFileSuffix=devFileSuffix,
                                        testFileSuffix=testFileSuffix,
                                        )

    if binary:
        binary_threshold = getBinaryThreshold(binary_threshold, y_train)
        y_train, y_dev, y_test = binarize(binary_threshold,
                                          y_train, y_dev, y_test)

    if pretrain_for:
        _extendVocabFor(
                    workspaceDir,
                    pretrain_for,
                    srcVocabTransformer,
                    refVocabTransformer,
                    devFileSuffix=pretrain_devFileSuffix,
                    testFileSuffix=pretrain_testFileSuffix,
        )

    model = getEnsembledModel(binary=binary,
                              srcVocabTransformer=srcVocabTransformer,
                              refVocabTransformer=refVocabTransformer,
                              **kwargs)

    if pretrained_model:
        logger.info("Loading weights into model")
        model.set_weights(model_weights)

    logger.info("Training model")

    if early_stop < 0:
        early_stop = epochs
    early_stop_monitor = ("val_acc" if binary else "val_pearsonr")

    model.fit_generator(getBatchGenerator([
                X_train['src'],
                X_train['mt']
            ], [
                y_train
            ],
            key=lambda x: "_".join(map(str, map(len, x))),
            batch_size=batchSize
        ),
        epochs=epochs,
        shuffle=True,
        validation_data=getBatchGenerator([
                X_dev['src'],
                X_dev['mt']
            ], [
                y_dev
            ],
            key=lambda x: "_".join(map(str, map(len, x)))
        ),
        callbacks=[
            EarlyStopping(monitor=early_stop_monitor,
                          patience=early_stop,
                          mode="max"),
        ],
        verbose=2
    )

    if saveModel:
        logger.info("Saving model")
        shelf = shelve.open(os.path.join(workspaceDir, "model." + saveModel))

        shelf['config'] = model.get_config()
        shelf['weights'] = model.get_weights()
        shelf['params'] = {
            'srcVocabTransformer': srcVocabTransformer,
            'refVocabTransformer': refVocabTransformer,
            'binary_threshold': binary_threshold,
        }

        shelf.close()

    logger.info("Evaluating on development data of size %d" % len(y_dev))
    dev_batches = getBatchGenerator([
            X_dev['src'],
            X_dev['mt']
        ],
        key=lambda x: "_".join(map(str, map(len, x)))
    )
    y_dev = dev_batches.align(y_dev)
    evaluate(
        model.predict_generator(dev_batches).reshape((-1,)),
        y_dev,
        binary=binary,
    )

    logger.info("Evaluating on test data of size %d" % len(y_test))
    test_batches = getBatchGenerator([
            X_test['src'],
            X_test['mt']
        ],
        key=lambda x: "_".join(map(str, map(len, x)))
    )
    y_test = test_batches.align(y_test)
    evaluate(
        model.predict_generator(test_batches).reshape((-1,)),
        y_test,
        binary=binary,
    )


def load_predictor(workspaceDir, saveModel,
                   max_len, num_buckets,
                   binary, binary_threshold,
                   **kwargs):
    shelf = shelve.open(os.path.join(workspaceDir, "model." + saveModel), 'r')

    srcVocabTransformer = shelf['params']['srcVocabTransformer']
    refVocabTransformer = shelf['params']['refVocabTransformer']

    binary_threshold = shelf['params']['binary_threshold']

    kwargs['src_fastText'] = None
    kwargs['ref_fastText'] = None

    model = getEnsembledModel(binary=binary,
                              srcVocabTransformer=srcVocabTransformer,
                              refVocabTransformer=refVocabTransformer,
                              **kwargs)

    logger.info("Loading weights into model")
    model.set_weights(shelf['weights'])

    shelf.close()

    def predictor(src, mt, y_test=None):
        logger.info("Preparing data for prediction")
        src = _preprocessSentences(src)
        mt = _preprocessSentences(mt)

        src = srcVocabTransformer.transform(src)
        mt = refVocabTransformer.transform(mt)

        srcMaxLen = min(max(map(len, src)), max_len)
        refMaxLen = min(max(map(len, mt)), max_len)

        src = pad_sequences(src, maxlen=srcMaxLen, num_buckets=num_buckets)
        mt = pad_sequences(mt, maxlen=refMaxLen, num_buckets=num_buckets)

        logger.info("Predicting")
        predict_batches = getBatchGenerator(
            [src, mt],
            key=lambda x: "_".join(map(str, map(len, x)))
        )

        predicted = model.predict_generator(predict_batches).reshape((-1,))

        predicted = predict_batches.alignOriginal(predicted)

        if y_test is not None:
            logger.info("Evaluating on test data of size %d" % len(y_test))
            if binary:
                y_test, = binarize(binary_threshold, y_test)

            evaluate(predicted, y_test, binary=binary)

        return predicted

    return predictor


def train(args):
    train_model(args.workspace_dir,
                args.data_name,
                devFileSuffix=args.dev_file_suffix,
                testFileSuffix=args.test_file_suffix,
                saveModel=args.save_model,
                batchSize=args.batch_size,
                epochs=args.epochs,
                early_stop=args.early_stop,
                ensemble_count=args.ensemble_count,

                pretrain_for=args.pretrain_for,
                pretrain_devFileSuffix=args.pretrain_dev_file_suffix,
                pretrain_testFileSuffix=args.pretrain_test_file_suffix,

                pretrained_model=args.pretrained_model,

                vocab_size=args.vocab_size,
                max_len=args.max_len,
                num_buckets=args.buckets,
                embedding_size=args.embedding_size,

                binary=args.binary,
                binary_threshold=args.binary_threshold,

                gru_size=args.gru_size,
                src_fastText=args.source_embeddings,
                ref_fastText=args.target_embeddings,
                train_embeddings=args.train_embeddings,
                attention=args.with_attention,
                summary_attention=args.summary_attention,
                use_estimator=(not args.no_estimator),
                )


def getPredictor(args):
    return load_predictor(args.workspace_dir,
                          saveModel=args.save_model,
                          ensemble_count=args.ensemble_count,
                          max_len=args.max_len,
                          num_buckets=args.buckets,
                          embedding_size=args.embedding_size,

                          binary=args.binary,
                          binary_threshold=args.binary_threshold,

                          gru_size=args.gru_size,
                          src_fastText=args.source_embeddings,
                          ref_fastText=args.target_embeddings,
                          train_embeddings=args.train_embeddings,
                          attention=args.with_attention,
                          summary_attention=args.summary_attention,
                          use_estimator=(not args.no_estimator),
                          )
