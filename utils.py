import numpy as np
import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def average_precision(pred, label):
    """calculate average precision
    for each relevant label, average precision computes the proportion
    of relevant labels that are ranked before it, and finally averages
    over all relevant labels [1]

    References:
    ----------
    .. [1] Sorower, Mohammad S. "A literature survey on algorithms for
    multi-label learning." Oregon State University, Corvallis (2010).

    Notes:
    -----
    .. The average_precision_score method in the sklearn.metrics package
    would produce a different numbers???

    average_precision_score(pred, label, average='samples')

    """
    ap = 0
    # sort the prediction scores in the descending order
    sorted_pred_idx = np.argsort(pred)[::-1]
    ranks = np.empty(len(pred), dtype=int)
    ranks[sorted_pred_idx] = np.arange(len(pred)) + 1

    # only care of those ranks of relevant labels
    ranks = ranks[label > 0]

    for ii, rank in enumerate(sorted(ranks)):
        num_relevant_labels = ii + 1  # including the current relevant label
        ap = ap + float(num_relevant_labels) / rank

    return 0 if len(ranks) == 0 else ap / len(ranks)


class AverageScore(object):
    """Compute precision/recall/f-score and mAP"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.threshold_values = list(np.arange(0.1, 1, 0.1))
        self.num_correct = [0] * len(self.threshold_values)
        self.num_pred = [0] * len(self.threshold_values)
        self.num_gold = 0
        self.num_samples = 0
        self.sum_ap = 0

    def update(self, preds, labels):
        batch_size = preds.shape[0]

        self.num_samples += batch_size
        ap = 0
        for i in range(batch_size):
            pred = preds[i]
            label = labels[i]

            correct_pred = pred[label > 0]
            self.num_gold = self.num_gold + len(np.nonzero(label)[0])

            for j, t in enumerate(self.threshold_values):
                self.num_pred[j] = self.num_pred[j] + len(pred[pred > t])
                self.num_correct[j] = self.num_correct[
                    j] + len(correct_pred[correct_pred > t])

            ap += average_precision(pred, label)

        self.sum_ap += ap

    def __str__(self):
        """String representation for logging
        """
        out = ''
        for i, t in enumerate(self.threshold_values):
            p = 0 if self.num_pred[i] == 0 else float(
                self.num_correct[i]) / self.num_pred[i]
            r = 0 if self.num_gold == 0 else float(
                self.num_correct[i]) / self.num_gold
            f = 0 if p + r == 0 else 2 * p * r / (p + r)
            out += '===> Precision = %.4f, Recall = %.4f, F-score = %.4f (@ threshold = %.1f)\n' % (
                p, r, f, t)
        out += '===> Mean AP = %.4f' % (self.sum_ap / self.num_samples)
        return out


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)
