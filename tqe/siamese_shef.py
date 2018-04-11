import os
import shelve

from keras.layers import dot, average, concatenate
from keras.layers import Input, Embedding
from keras.layers import Dense
from keras.layers import Conv1D, GlobalMaxPooling1D, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping

# from keras.utils.generic_utils import CustomObjectScope

from .common import evaluate

from .common import _prepareInput, _extendVocabFor
from .common import WordIndexTransformer
from .common import _printModelSummary
from .common import getBatchGenerator
from .common import pearsonr
from .common import get_fastText_embeddings
from .common import _preprocessSentences, pad_sequences

from .baseline import _loadAndPrepareFeatures, _prepareFeatures

import logging
logger = logging.getLogger("siamese-shef")


def getSentenceEncoder(vocabTransformer,
                       embedding_size,
                       fastText,
                       train_embeddings,
                       filter_sizes, num_filters, sentence_vector_size,
                       cnn_dropout,
                       model_inputs, verbose,
                       ):
    vocab_size = vocabTransformer.vocab_size()

    embedding_kwargs = {}

    if fastText:
        if verbose:
            logger.info("Loading fastText embeddings from: " + fastText)
        embedding_kwargs['weights'] = [get_fastText_embeddings(
                                fastText,
                                vocabTransformer,
                                embedding_size
                                )]

    input, = model_inputs

    embedding = Embedding(
                        output_dim=embedding_size,
                        input_dim=vocab_size,
                        name="embedding",
                        trainable=train_embeddings,
                        **embedding_kwargs)(input)

    conv_blocks = []
    for filter_size in filter_sizes:
        conv = Conv1D(
                    filters=num_filters,
                    kernel_size=filter_size,
                    name="conv_" + str(filter_size),
                )(embedding)
        conv = GlobalMaxPooling1D(
                    name="global_maxpool_" + str(filter_size)
                )(conv)
        conv_blocks.append(conv)

    z = concatenate(conv_blocks, name="concat") \
        if len(conv_blocks) > 1 else conv_blocks[0]

    if cnn_dropout > 0:
        z = Dropout(cnn_dropout, name="cnn_dropout")(z)

    encoder = Dense(sentence_vector_size, name="sentence_enc")(z)

    sentence_encoder = Model(inputs=input, outputs=encoder)

    if verbose:
        _printModelSummary(logger, sentence_encoder, "sentence_encoder")

    return sentence_encoder


def getFeaturesEncoder(model_inputs, mlp_size, verbose):
    features_input, = model_inputs

    hidden = Dense(mlp_size, activation="tanh")(features_input)
    encoder = Dense(mlp_size, activation="tanh")(hidden)

    features_encoder = Model(inputs=model_inputs, outputs=encoder)

    if verbose:
        _printModelSummary(logger, features_encoder, "feature_encoder")

    return features_encoder


def getModel(srcVocabTransformer, refVocabTransformer,
             src_fastText, ref_fastText,
             num_features,
             mlp_size,
             use_siamese, use_shef,
             model_inputs=None, verbose=False,
             **kwargs
             ):
    if verbose:
        logger.info("Creating model")

    if not model_inputs:
        model_inputs = [
            Input(shape=(None, ), name="input_src"),
            Input(shape=(None, ), name="input_mt")
        ]

        if num_features:
            model_inputs = [
                Input(shape=(num_features, ), name="input_features")
            ] + model_inputs

    if num_features:
        features_input, src_input, ref_input = model_inputs
    else:
        src_input, ref_input = model_inputs

    src_sentence_enc = getSentenceEncoder(vocabTransformer=srcVocabTransformer,
                                          fastText=src_fastText,
                                          model_inputs=[src_input],
                                          verbose=verbose,
                                          **kwargs)(src_input)

    ref_sentence_enc = getSentenceEncoder(vocabTransformer=refVocabTransformer,
                                          fastText=ref_fastText,
                                          model_inputs=[ref_input],
                                          verbose=verbose,
                                          **kwargs)(ref_input)

    if use_shef:
        encodings = [src_sentence_enc, ref_sentence_enc]
        if num_features:
            features_enc = getFeaturesEncoder(mlp_size=mlp_size,
                                              model_inputs=[features_input],
                                              verbose=verbose,
                                              )(features_input)
            encodings = [features_enc] + encodings

        hidden = concatenate(encodings)

        hidden = Dense(mlp_size, activation="tanh")(hidden)
        hidden = Dense(mlp_size, activation="tanh")(hidden)

    if use_siamese and use_shef:
        siamese_quality = dot([src_sentence_enc, ref_sentence_enc],
                              axes=-1,
                              normalize=True,
                              name="siamese_quality")

        shef_quality = Dense(1, name="shef_quality",
                             activation="sigmoid")(hidden)

        quality = average([siamese_quality, shef_quality], name="quality")
    elif use_siamese:
        quality = dot([src_sentence_enc, ref_sentence_enc],
                      axes=-1,
                      normalize=True,
                      name="quality")
    elif use_shef:
        quality = Dense(1, name="quality",
                        activation="sigmoid")(hidden)
    else:
        raise ValueError("Please specify atleast one of `Siamese` or `SHEF` "
                         "model to use.")

    if verbose:
        logger.info("Compiling model")
    model = Model(inputs=model_inputs,
                  outputs=[quality])

    model.compile(
            optimizer="adadelta",
            loss={
                "quality": "mse"
            },
            metrics={
                "quality": ["mse", "mae", pearsonr]
            }
        )
    if verbose:
        _printModelSummary(logger, model, "model")

    return model


def getEnsembledModel(ensemble_count, num_features, **kwargs):
    kwargs['num_features'] = num_features

    if ensemble_count == 1:
        return getModel(verbose=True, **kwargs)

    model_inputs = [
        Input(shape=(None, ), name="input_src"),
        Input(shape=(None, ), name="input_mt")
    ]

    if num_features:
        model_inputs = [
            Input(shape=(num_features, ), name="input_features")
        ] + model_inputs

    logger.info("Creating models to ensemble")
    verbose = [True] + [False] * (ensemble_count - 1)
    models = [getModel(model_inputs=model_inputs, verbose=v, **kwargs)
              for v in verbose]

    output = average([model(model_inputs) for model in models],
                     name='quality')

    logger.info("Compiling ensembled model")
    model = Model(inputs=model_inputs,
                  outputs=output)

    model.compile(
            optimizer="adadelta",
            loss={
                "quality": "mse"
            },
            metrics={
                "quality": ["mse", "mae", pearsonr]
            }
        )
    _printModelSummary(logger, model, "ensembled_model")

    return model


def _get_embedding_path(workspaceDir, model):
    return os.path.join(workspaceDir,
                        "fastText",
                        ".".join([model, "bin"])
                        ) if model else None


def train_model(workspaceDir, dataName, devFileSuffix, testFileSuffix,
                pretrain_for,
                pretrain_devFileSuffix, pretrain_testFileSuffix,
                pretrained_model,
                saveModel,
                use_features,
                featureFileSuffix, normalize, trainLM, trainNGrams,
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

        # Setup feature extraction params
        pretrainedBasename = os.path.join(workspaceDir,
                                          "tqe." + shelf['args'].data_name)
        standardScaler = shelf['params']['standardScaler']

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

        # Setup feature extraction params
        pretrainedBasename = None
        standardScaler = None

        # Set paths for fastText models
        kwargs['src_fastText'] = _get_embedding_path(workspaceDir,
                                                     kwargs['src_fastText'])
        kwargs['ref_fastText'] = _get_embedding_path(workspaceDir,
                                                     kwargs['ref_fastText'])

    X_train, y_train, X_dev, y_dev, X_test, y_test = _prepareInput(
                                        workspaceDir,
                                        dataName,
                                        srcVocabTransformer,
                                        refVocabTransformer,
                                        train_vocab=train_vocab,
                                        max_len=max_len,
                                        num_buckets=num_buckets,
                                        devFileSuffix=devFileSuffix,
                                        testFileSuffix=testFileSuffix,
                                        )

    if pretrain_for:
        _extendVocabFor(
                    workspaceDir,
                    pretrain_for,
                    srcVocabTransformer,
                    refVocabTransformer,
                    devFileSuffix=pretrain_devFileSuffix,
                    testFileSuffix=pretrain_testFileSuffix,
        )

    if use_features:
        (standardScaler,
         (X_train['features'], _,
          X_dev['features'], _,
          X_test['features'], _)) = \
            _loadAndPrepareFeatures(
                os.path.join(workspaceDir, "tqe." + dataName),
                trainedBasename=pretrainedBasename,
                standardScaler=standardScaler,
                devFileSuffix=devFileSuffix, testFileSuffix=testFileSuffix,
                featureFileSuffix=featureFileSuffix,
                normalize=normalize,
                trainLM=trainLM,
                trainNGrams=trainNGrams,
            )

        num_features = len(X_train['features'][0])
        inputs = ['features', 'src', 'mt']
    else:
        standardScaler = None
        num_features = 0
        inputs = ['src', 'mt']

    model = getEnsembledModel(num_features=num_features,
                              srcVocabTransformer=srcVocabTransformer,
                              refVocabTransformer=refVocabTransformer,
                              **kwargs)

    if pretrained_model:
        logger.info("Loading weights into model")
        model.set_weights(model_weights)

    logger.info("Training model")

    if early_stop < 0:
        early_stop = epochs

    model.fit_generator(getBatchGenerator(
            map(lambda input: X_train[input], inputs),
            [
                y_train
            ],
            key=lambda x: "_".join(map(str, map(len, x))),
            batch_size=batchSize
        ),
        epochs=epochs,
        shuffle=True,
        validation_data=getBatchGenerator(
            map(lambda input: X_dev[input], inputs),
            [
                y_dev
            ],
            key=lambda x: "_".join(map(str, map(len, x)))
        ),
        callbacks=[
            EarlyStopping(monitor="val_pearsonr", patience=early_stop,
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
            'standardScaler': standardScaler,
            'num_features': num_features,
        }

        shelf.close()

    logger.info("Evaluating on development data of size %d" % len(y_dev))
    dev_batches = getBatchGenerator(
        map(lambda input: X_dev[input], inputs),
        key=lambda x: "_".join(map(str, map(len, x))),
        batch_size=batchSize
    )
    y_dev = dev_batches.align(y_dev)
    evaluate(
        model.predict_generator(dev_batches).reshape((-1,)),
        y_dev
    )

    logger.info("Evaluating on test data of size %d" % len(y_test))
    test_batches = getBatchGenerator(
        map(lambda input: X_test[input], inputs),
        key=lambda x: "_".join(map(str, map(len, x))),
        batch_size=batchSize
    )
    y_test = test_batches.align(y_test)
    evaluate(
        model.predict_generator(test_batches).reshape((-1,)),
        y_test
    )


def load_predictor(workspaceDir, dataName, saveModel,
                   max_len, num_buckets,
                   use_features,
                   **kwargs):
    shelf = shelve.open(os.path.join(workspaceDir, "model." + saveModel), 'r')

    srcVocabTransformer = shelf['params']['srcVocabTransformer']
    refVocabTransformer = shelf['params']['refVocabTransformer']

    standardScaler = shelf['params']['standardScaler']
    num_features = shelf['params']['num_features']

    kwargs['src_fastText'] = None
    kwargs['ref_fastText'] = None

    model = getEnsembledModel(num_features=num_features,
                              srcVocabTransformer=srcVocabTransformer,
                              refVocabTransformer=refVocabTransformer,
                              **kwargs)

    logger.info("Loading weights into model")
    model.set_weights(shelf['weights'])

    shelf.close()

    def _prepareSentences(sentences, vocavTrannsformer):
        sentences = _preprocessSentences(sentences)
        sentences = vocavTrannsformer.transform(sentences)
        currMaxLen = min(max(map(len, sentences)), max_len)

        return pad_sequences(sentences, maxlen=currMaxLen,
                             num_buckets=num_buckets)

    def predictor(src, mt, y_test=None):
        logger.info("Preparing data for prediction")

        inputs = [_prepareSentences(src, srcVocabTransformer),
                  _prepareSentences(mt, refVocabTransformer)]

        if use_features:
            srcSentences = _preprocessSentences(src,
                                                lower=False, tokenize=False)
            mtSentences = _preprocessSentences(mt,
                                               lower=False, tokenize=False)

            features, = _prepareFeatures(
                                os.path.join(workspaceDir, "tqe." + dataName),
                                [{"src": srcSentences, "mt": mtSentences}]
                            )

            if standardScaler:
                features = standardScaler.transform(features)

            inputs = [features] + inputs

        logger.info("Predicting")
        predict_batches = getBatchGenerator(
            inputs,
            key=lambda x: "_".join(map(str, map(len, x)))
        )

        predicted = model.predict_generator(predict_batches).reshape((-1,))

        predicted = predict_batches.alignOriginal(predicted)

        if y_test is not None:
            logger.info("Evaluating on test data of size %d" % len(y_test))
            evaluate(predicted, y_test)

        return predicted

    return predictor


def validateArgs(args):
    if not args.use_shef and not args.use_siamese:
        raise ValueError("Cannot disable both Siamese and SHEF networks")

    if not args.use_shef:
        args.use_features = False

    return args


def train(args):
    args = validateArgs(args)
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

                use_features=args.use_features,
                use_siamese=args.use_siamese,
                use_shef=args.use_shef,

                mlp_size=args.mlp_size,

                vocab_size=args.vocab_size,
                max_len=args.max_len,
                num_buckets=args.buckets,
                embedding_size=args.embedding_size,
                src_fastText=args.source_embeddings,
                ref_fastText=args.target_embeddings,
                train_embeddings=args.train_embeddings,
                filter_sizes=args.filter_sizes,
                num_filters=args.num_filters,
                sentence_vector_size=args.sentence_vector_size,
                cnn_dropout=args.cnn_dropout,

                featureFileSuffix=args.feature_file_suffix,
                normalize=args.normalize,
                trainLM=args.train_lm,
                trainNGrams=args.train_ngrams,
                )


def getPredictor(args):
    args = validateArgs(args)
    return load_predictor(args.workspace_dir,
                          args.data_name,
                          saveModel=args.save_model,
                          ensemble_count=args.ensemble_count,

                          use_features=args.use_features,
                          use_siamese=args.use_siamese,
                          use_shef=args.use_shef,

                          mlp_size=args.mlp_size,

                          max_len=args.max_len,
                          num_buckets=args.buckets,
                          embedding_size=args.embedding_size,
                          src_fastText=args.source_embeddings,
                          ref_fastText=args.target_embeddings,
                          train_embeddings=args.train_embeddings,
                          filter_sizes=args.filter_sizes,
                          num_filters=args.num_filters,
                          sentence_vector_size=args.sentence_vector_size,
                          cnn_dropout=args.cnn_dropout,
                          )
