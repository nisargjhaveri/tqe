import argparse
import os
import shelve


def setupSubparsers(parser):
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('workspace_dir',
                               help='Directory containing prepared files')
    common_parser.add_argument('--save-model', type=str, default=None,
                               help='Save the model with this basename')
    common_parser.add_argument('data_name',
                               help='Identifier for prepared files')
    common_parser.add_argument('--dev-file-suffix', type=str, default=None,
                               help='Suffix for dev files')
    common_parser.add_argument('--test-file-suffix', type=str, default=None,
                               help='Suffix for test files')

    subparsers = parser.add_subparsers(title='subcommands', dest="model")

    predict_parser = subparsers.add_parser('predict')
    predict_parser.add_argument('workspace_dir',
                                help='Directory containing prepared files')
    predict_parser.add_argument('model_name', type=str,
                                help="Basepath of saved model")
    predict_parser.add_argument('data_name',
                                help='Identifier for prepared files')
    predict_parser.add_argument('--data-file-suffix', type=str, default=None,
                                help='Suffix for data file')
    predict_parser.add_argument('--evaluate', action='store_true',
                                help="Evaluate along with prediction.")
    predict_parser.add_argument('--print-results', action='store_true',
                                help="Print result")
    predict_parser.add_argument('--save-results', type=str, default=None,
                                help="Save results to file")

    parent = {'parents': [common_parser]}
    baselineArgparser(subparsers.add_parser('baseline', **parent))
    postechArgparser(subparsers.add_parser('postech', **parent))
    rnnArgparser(subparsers.add_parser('rnn', **parent))
    siameseArgparser(subparsers.add_parser('siamese', **parent))
    shefArgparser(subparsers.add_parser('shef', **parent))
    siameseShefArgparser(subparsers.add_parser('siamese-shef', **parent))


def _getModel(model):
    if model == "baseline":
        from tqe import baseline
        return baseline
    elif model == "postech":
        from tqe import postech
        return postech
    elif model == "rnn":
        from tqe import rnn
        return rnn
    elif model == "siamese":
        from tqe import siamese
        return siamese
    elif model == "shef":
        from tqe import shef
        return shef
    elif model == "siamese-shef":
        from tqe import siamese_shef
        return siamese_shef
    else:
        raise RuntimeError("Unknown model name")


def run(args):
    if (args.model == 'predict'):
        import numpy as np

        filenameSuffix = ""
        if args.data_file_suffix:
            filenameSuffix = args.data_file_suffix

        fileBasename = os.path.join(args.workspace_dir,
                                    "tqe." + args.data_name)

        srcSentencesPath = fileBasename + ".src" + filenameSuffix
        mtSentencesPath = fileBasename + ".mt" + filenameSuffix

        with open(srcSentencesPath) as lines:
            src = map(lambda l: l.decode('utf-8'), list(lines))
        with open(mtSentencesPath) as lines:
            mt = map(lambda l: l.decode('utf-8'), list(lines))

        y = None
        if args.evaluate:
            targetPath = fileBasename + ".hter" + filenameSuffix
            y = np.clip(np.loadtxt(targetPath), 0, 1)

        modelPath = os.path.join(args.workspace_dir, args.model_name)
        predicted = getPredictor(modelPath)(src, mt, y)

        if args.print_results:
            print "\n".join(list(predicted))

        if args.save_results:
            np.savetxt(os.path.join(args.workspace_dir, args.save_results),
                       predicted,
                       fmt="%.8f")
    else:
        train(args)


def train(args):
    if args.save_model:
        shelf = shelve.open(os.path.join(args.workspace_dir,
                                         "model." + args.save_model))

        shelf['args'] = args

        shelf.close()

    _getModel(args.model).train(args)


def getPredictor(modelPath):
    if not modelPath:
        raise ValueError("Model path must be specified to load predictor")

    shelf = shelve.open(modelPath, 'r')
    args = shelf['args']
    shelf.close()

    return _getModel(args.model).getPredictor(args)


def baselineArgparser(parser):
    parser.add_argument('--feature-file-suffix', type=str, default=None,
                        help='Suffix for feature files')

    parser.add_argument('--pretrained-model', type=str, default=None,
                        help='Pre-trained model to use for feature extraction')
    parser.add_argument('--train-lm', action='store_true',
                        help='Train language model.')
    parser.add_argument('--train-ngrams', action='store_true',
                        help='Compute ngram freqs.')
    parser.add_argument('--normalize', action='store_true',
                        help='Weather to normalize features or not.')
    parser.add_argument('--tune', action='store_true',
                        help='Weather to tune parameters or not.')
    parser.add_argument('--max-jobs', type=int, default=-1,
                        help='Maximum number of jobs to run parallelly')


def postechArgparser(parser):
    parser.add_argument('-b', '--batch-size', type=int, default=50,
                        help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=25,
                        help='Number of epochs to run')
    parser.add_argument('--early-stop', type=int, default=2,
                        help="Patience before early stopping. "
                        "-1 for no early stopping")
    parser.add_argument('--ensemble-count', type=int, default=1,
                        help='Number of models to ensemble')

    parser.add_argument('--max-len', type=int, default=100,
                        help='Maximum length of the sentences')
    parser.add_argument('--buckets', type=int, default=4,
                        help='Number of buckets for padding lenght')
    parser.add_argument('-m', '--embedding-size', type=int, default=300,
                        help='Size of word embeddings')

    parser.add_argument('-n', '--gru-size', type=int, default=500,
                        help='Size of GRU')
    parser.add_argument('-q', '--qualvec-size', type=int, default=500,
                        help='Size of last layer connected before softmax')
    parser.add_argument('-l', '--maxout-size', type=int, default=500,
                        help='Size of maxout layer output')
    parser.add_argument('-v', '--vocab-size', type=int, default=40000,
                        help='Maximum vocab size')
    parser.add_argument('--maxout-units', type=int, default=2,
                        help='Number of maxout units')

    parser.add_argument('--predictor-data', type=str, default=None,
                        help='Identifier for data to pre-train predictor with')
    parser.add_argument('--predictor-model', type=str, default=None,
                        help='Name of predictor model to save/load from')

    parser.add_argument('--two-step', action="store_true",
                        help='Use two step training instead of multitask')
    parser.add_argument('--stack-prop', action="store_true",
                        help='Use stack propogation instead of multitask')


def rnnArgparser(parser):
    parser.add_argument('--pretrain-for', type=str, default=None,
                        help='Pre-train for data specified')
    parser.add_argument('--pretrain-dev-file-suffix', type=str, default=None,
                        help='Suffix for dev files')
    parser.add_argument('--pretrain-test-file-suffix', type=str, default=None,
                        help='Suffix for test files')

    parser.add_argument('--pretrained-model', type=str, default=None,
                        help='Pre-trained model to load and continue '
                        'training.')

    parser.add_argument('-b', '--batch-size', type=int, default=50,
                        help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=25,
                        help='Number of epochs to run')
    parser.add_argument('--early-stop', type=int, default=2,
                        help="Patience before early stopping. "
                        "-1 for no early stopping")
    parser.add_argument('--ensemble-count', type=int, default=3,
                        help='Number of models to ensemble')
    parser.add_argument('--max-len', type=int, default=100,
                        help='Maximum length of the sentences')
    parser.add_argument('--buckets', type=int, default=4,
                        help='Number of buckets for padding lenght')

    parser.add_argument('--binary', action='store_true',
                        help='Train for binary classification.')
    parser.add_argument('--binary-threshold', type=float, default=None,
                        help='Threshold for binary classification.')

    parser.add_argument('--source-embeddings', type=str, default=None,
                        help='fastText model name for target language')
    parser.add_argument('--target-embeddings', type=str, default=None,
                        help='fastText model name for target language')
    parser.add_argument('--freeze-embeddings', action="store_false",
                        dest="train_embeddings",
                        help="Make embedding layers non-trainable")
    parser.add_argument('-m', '--embedding-size', type=int, default=300,
                        help='Size of word embeddings')
    parser.add_argument('-n', '--gru-size', type=int, default=500,
                        help='Size of GRU')
    parser.add_argument('-v', '--vocab-size', type=int, default=40000,
                        help='Maximum vocab size')
    parser.add_argument('--with-attention', action="store_true",
                        help='Add attention in decoder')
    parser.add_argument('--summary-attention', action="store_true",
                        help='Get quality summary using attention')
    parser.add_argument('--no-estimator', action="store_true",
                        help='Don\'t use separate estimator layer')


def siameseArgparser(parser):
    parser.add_argument('-b', '--batch-size', type=int, default=50,
                        help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=25,
                        help='Number of epochs to run')
    parser.add_argument('--ensemble-count', type=int, default=3,
                        help='Number of models to ensemble')
    parser.add_argument('--max-len', type=int, default=100,
                        help='Maximum length of the sentences')
    parser.add_argument('--buckets', type=int, default=4,
                        help='Number of buckets for padding lenght')
    parser.add_argument('--source-embeddings', type=str, default=None,
                        help='fastText model name for target language')
    parser.add_argument('--target-embeddings', type=str, default=None,
                        help='fastText model name for target language')
    parser.add_argument('-v', '--vocab-size', type=int, default=40000,
                        help='Maximum vocab size')
    parser.add_argument('-m', '--embedding-size', type=int, default=300,
                        help='Size of word embeddings')

    parser.add_argument('--filter-sizes', type=int, nargs='*',
                        default=[3, 4, 5],
                        help='Filter sizes')
    parser.add_argument('--num-filters', type=int, default=200,
                        help='Number of filters for each sizes')
    parser.add_argument('--sentence-vector-size', type=int, default=500,
                        help='Size of sentence vector')
    parser.add_argument('--cnn-dropout', type=float, default=0,
                        help='Dropout in CNN encoder')


def shefArgparser(parser):
    parser.add_argument('-b', '--batch-size', type=int, default=50,
                        help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='Number of epochs to run')
    parser.add_argument('--no-early-stop', action="store_false",
                        dest="early_stop",
                        help="Don't early stop using validation")
    parser.add_argument('--ensemble-count', type=int, default=1,
                        help='Number of models to ensemble')

    parser.add_argument('--max-len', type=int, default=500,
                        help='Maximum length of the sentences')
    parser.add_argument('-v', '--vocab-size', type=int, default=40000,
                        help='Maximum vocab size')
    parser.add_argument('-m', '--embedding-size', type=int, default=50,
                        help='Size of word embeddings')

    parser.add_argument('--no-features', action="store_false",
                        dest="use_features",
                        help="Don't use manual features")
    parser.add_argument('--filter-sizes', type=int, nargs='*',
                        default=[1, 2, 3, 4],
                        help='Filter sizes')
    parser.add_argument('--num-filters', type=int, default=50,
                        help='Number of filters for each sizes')
    parser.add_argument('--pool-length', type=int, default=4,
                        help='Max-pool window size')
    parser.add_argument('--cnn-depth', type=int, default=4,
                        help='How many convolution/max-pooling to stack')
    parser.add_argument('--cnn-dropout', type=float, default=0,
                        help='Dropout in CNN encoder')

    parser.add_argument('--feature-file-suffix', type=str, default=None,
                        help='Suffix for feature files')
    parser.add_argument('--train-lm', action='store_true',
                        help='Train language model.')
    parser.add_argument('--train-ngrams', action='store_true',
                        help='Compute ngram freqs.')
    parser.add_argument('--normalize', action='store_true',
                        help='Weather to normalize features or not.')


def siameseShefArgparser(parser):
    parser.add_argument('--pretrain-for', type=str, default=None,
                        help='Pre-train for data specified')
    parser.add_argument('--pretrain-dev-file-suffix', type=str, default=None,
                        help='Suffix for dev files')
    parser.add_argument('--pretrain-test-file-suffix', type=str, default=None,
                        help='Suffix for test files')

    parser.add_argument('--pretrained-model', type=str, default=None,
                        help='Pre-trained model to load and continue '
                        'training.')

    parser.add_argument('-b', '--batch-size', type=int, default=50,
                        help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=25,
                        help='Number of epochs to run')
    parser.add_argument('--early-stop', type=int, default=2,
                        help="Patience before early stopping. "
                        "-1 for no early stopping")
    parser.add_argument('--ensemble-count', type=int, default=3,
                        help='Number of models to ensemble')

    parser.add_argument('--max-len', type=int, default=100,
                        help='Maximum length of the sentences')
    parser.add_argument('--buckets', type=int, default=4,
                        help='Number of buckets for padding lenght')
    parser.add_argument('--source-embeddings', type=str, default=None,
                        help='fastText model name for target language')
    parser.add_argument('--target-embeddings', type=str, default=None,
                        help='fastText model name for target language')
    parser.add_argument('--freeze-embeddings', action="store_false",
                        dest="train_embeddings",
                        help="Make embedding layers non-trainable")
    parser.add_argument('-v', '--vocab-size', type=int, default=40000,
                        help='Maximum vocab size')
    parser.add_argument('-m', '--embedding-size', type=int, default=300,
                        help='Size of word embeddings')

    parser.add_argument('--binary', action='store_true',
                        help='Train for binary classification.')
    parser.add_argument('--binary-threshold', type=float, default=None,
                        help='Threshold for binary classification.')

    parser.add_argument('--no-features', action="store_false",
                        dest="use_features",
                        help="Don't use manual features")
    parser.add_argument('--no-siamese', action="store_false",
                        dest="use_siamese",
                        help="Don't use siamese network")
    parser.add_argument('--no-shef', action="store_false",
                        dest="use_shef",
                        help="Don't use SHEF network")

    parser.add_argument('-n', '--mlp-size', type=int, default=50,
                        help='Number of hidden units in MLP layers')

    parser.add_argument('--filter-sizes', type=int, nargs='*',
                        default=[3, 4, 5],
                        help='Filter sizes')
    parser.add_argument('--num-filters', type=int, default=200,
                        help='Number of filters for each sizes')
    parser.add_argument('--sentence-vector-size', type=int, default=500,
                        help='Size of sentence vector')
    parser.add_argument('--cnn-dropout', type=float, default=0,
                        help='Dropout in CNN encoder')

    parser.add_argument('--feature-file-suffix', type=str, default=None,
                        help='Suffix for feature files')
    parser.add_argument('--train-lm', action='store_true',
                        help='Train language model.')
    parser.add_argument('--train-ngrams', action='store_true',
                        help='Compute ngram freqs.')
    parser.add_argument('--normalize', action='store_true',
                        help='Weather to normalize features or not.')
