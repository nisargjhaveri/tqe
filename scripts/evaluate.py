import argparse

import numpy as np
from sklearn.metrics import classification_report
# from sklearn.metrics import precision_recall_fscore_support


def bad_accuracy(y_pred, y_true):
    y_pred_bin = np.where(y_pred >= np.mean(y_pred), 1, 0)
    y_true_bin = np.where(y_true >= np.mean(y_true), 1, 0)

    # precision, recall, f1, support = \
    #     precision_recall_fscore_support(y_true_bin, y_pred_bin, average=None)
    #
    # avg_precision, avg_recall, avg_f1, total_support = \
    #     precision_recall_fscore_support(y_true_bin, y_pred_bin,
    #                                     average='weighted')

    print classification_report(y_true_bin, y_pred_bin,
                                [0, 1], ["good", "bad"])


def evaluate(y_pred, y_test, output=True):
    """
    Should always be identical to the function `evaluate` in `tqe/common.py`
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from scipy.stats import pearsonr
    from scipy.stats import spearmanr

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    pearsonR = pearsonr(y_pred, y_test)
    spearmanR = spearmanr(y_pred, y_test)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Evaluate quality predictions')

    parser.add_argument('pred_file', type=file,
                        help='Predicted labels file')
    parser.add_argument('true_file', type=file,
                        help='True labels file')

    parser.add_argument('-b', '--binary', action="store_true",
                        help='Evaluate binary classification accuracy')

    args = parser.parse_args()

    y_pred = np.clip(np.loadtxt(args.pred_file), 0, 1)
    y_true = np.clip(np.loadtxt(args.true_file), 0, 1)

    if (args.binary):
        bad_accuracy(y_pred, y_true)
    else:
        evaluate(y_pred, y_true)
