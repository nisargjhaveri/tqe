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

from .common import _prepareInput
from .common import WordIndexTransformer
from .common import _printModelSummary
from .common import getBatchGenerator
from .common import pearsonr
from .common import get_fastText_embeddings

from .baseline import _loadAndPrepareFeatures

import logging
logger = logging.getLogger("siamese-shef")


def getSentenceEncoder(vocabTransformer,
                       embedding_size,
                       fastText,
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
                        **embedding_kwargs)(input)

    conv_blocks = []
    for filter_size in filter_sizes:
        conv = Conv1D(
                    filters=num_filters,
                    kernel_size=filter_size
                )(embedding)
        conv = GlobalMaxPooling1D()(conv)
        conv_blocks.append(conv)

    z = concatenate(conv_blocks) \
        if len(conv_blocks) > 1 else conv_blocks[0]

    if cnn_dropout > 0:
        z = Dropout(cnn_dropout)(z)

    encoder = Dense(sentence_vector_size)(z)

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
             model_inputs=None, verbose=False,
             **kwargs
             ):
    if verbose:
        logger.info("Creating model")

    if not model_inputs:
        model_inputs = [
            Input(shape=(num_features, )),
            Input(shape=(None, )),
            Input(shape=(None, ))
        ]

    features_input, src_input, ref_input = model_inputs

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

    features_enc = getFeaturesEncoder(mlp_size=mlp_size,
                                      model_inputs=[features_input],
                                      verbose=verbose,
                                      )(features_input)

    siamese_quality = dot([src_sentence_enc, ref_sentence_enc],
                          axes=-1,
                          normalize=True,
                          name="siamese_quality")

    hidden = concatenate([features_enc, src_sentence_enc, ref_sentence_enc])

    hidden = Dense(mlp_size, activation="tanh")(hidden)
    hidden = Dense(mlp_size, activation="tanh")(hidden)

    shef_quality = Dense(1, name="shef_quality", activation="sigmoid")(hidden)

    quality = average([siamese_quality, shef_quality], name="quality")

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

    features_input = Input(shape=(num_features, ))
    src_input = Input(shape=(None, ))
    ref_input = Input(shape=(None, ))

    model_inputs = [features_input, src_input, ref_input]

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


def train_model(workspaceDir, modelName, devFileSuffix, testFileSuffix,
                saveModel,
                featureFileSuffix, normalize, trainLM, trainNGrams,
                batchSize, epochs, max_len, num_buckets, vocab_size,
                **kwargs):
    logger.info("initializing TQE training")

    srcVocabTransformer = WordIndexTransformer(vocab_size=vocab_size)
    refVocabTransformer = WordIndexTransformer(vocab_size=vocab_size)

    X_train, y_train, X_dev, y_dev, X_test, y_test = _prepareInput(
                                        workspaceDir,
                                        modelName,
                                        srcVocabTransformer,
                                        refVocabTransformer,
                                        max_len=max_len,
                                        num_buckets=num_buckets,
                                        devFileSuffix=devFileSuffix,
                                        testFileSuffix=testFileSuffix,
                                        )
    (standardScaler,
     (X_train['features'], _, X_dev['features'], _, X_test['features'], _)) = \
        _loadAndPrepareFeatures(
            os.path.join(workspaceDir, "tqe." + modelName),
            devFileSuffix=devFileSuffix, testFileSuffix=testFileSuffix,
            featureFileSuffix=featureFileSuffix,
            normalize=normalize,
            trainLM=trainLM,
            trainNGrams=trainNGrams,
        )

    num_features = len(X_train['features'][0])

    def get_embedding_path(model):
        return os.path.join(workspaceDir,
                            "fastText",
                            ".".join([model, "bin"])
                            ) if model else None

    kwargs['src_fastText'] = get_embedding_path(kwargs['src_fastText'])
    kwargs['ref_fastText'] = get_embedding_path(kwargs['ref_fastText'])

    model = getEnsembledModel(num_features=num_features,
                              srcVocabTransformer=srcVocabTransformer,
                              refVocabTransformer=refVocabTransformer,
                              **kwargs)

    logger.info("Training model")

    model.fit_generator(getBatchGenerator([
                X_train['features'],
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
                X_dev['features'],
                X_dev['src'],
                X_dev['mt']
            ], [
                y_dev
            ],
            key=lambda x: "_".join(map(str, map(len, x)))
        ),
        callbacks=[
            EarlyStopping(monitor="val_pearsonr", patience=2, mode="max"),
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
        }

        shelf.close()

    logger.info("Evaluating on development data of size %d" % len(y_dev))
    dev_batches = getBatchGenerator([
            X_dev['features'],
            X_dev['src'],
            X_dev['mt']
        ],
        key=lambda x: "_".join(map(str, map(len, x)))
    )
    y_dev = dev_batches.align(y_dev)
    evaluate(
        model.predict_generator(dev_batches).reshape((-1,)),
        y_dev
    )

    logger.info("Evaluating on test data of size %d" % len(y_test))
    test_batches = getBatchGenerator([
            X_test['features'],
            X_test['src'],
            X_test['mt']
        ],
        key=lambda x: "_".join(map(str, map(len, x)))
    )
    y_test = test_batches.align(y_test)
    evaluate(
        model.predict_generator(test_batches).reshape((-1,)),
        y_test
    )


def train(args):
    train_model(args.workspace_dir,
                args.data_name,
                devFileSuffix=args.dev_file_suffix,
                testFileSuffix=args.test_file_suffix,

                saveModel=args.save_model,
                batchSize=args.batch_size,
                epochs=args.epochs,
                ensemble_count=args.ensemble_count,

                mlp_size=args.mlp_size,

                vocab_size=args.vocab_size,
                max_len=args.max_len,
                num_buckets=args.buckets,
                embedding_size=args.embedding_size,
                src_fastText=args.source_embeddings,
                ref_fastText=args.target_embeddings,
                filter_sizes=args.filter_sizes,
                num_filters=args.num_filters,
                sentence_vector_size=args.sentence_vector_size,
                cnn_dropout=args.cnn_dropout,

                featureFileSuffix=args.feature_file_suffix,
                normalize=args.normalize,
                trainLM=args.train_lm,
                trainNGrams=args.train_ngrams,
                )
