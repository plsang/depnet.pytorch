import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import time

import logging
logger = logging.getLogger(__name__)


def train(opt, model, criterion, optimizer, train_loader, epoch):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, train_data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = Variable(train_data[0], volatile=False)
        labels = Variable(train_data[1], volatile=False)

        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        preds = model(images)
        #import pdb; pdb.set_trace()
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)

        # Print log info
        if i % opt.log_step == 0:
            logger.info(
                'Epoch [{0}][{1}/{2}]\t'
                'Loss {3:0.7f}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(
                    epoch, i, len(train_loader), loss.data[0],
                    batch_time=batch_time,
                    data_time=data_time))

        end = time.time()


def test(opt, model, criterion, val_loader):
    val_loss = AverageMeter()
    val_score = AverageScore()

    model.eval()
    for i, val_data in enumerate(val_loader):
        # Update the model
        images = Variable(val_data[0], volatile=True)
        labels = Variable(val_data[1], volatile=True)

        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        preds = model(images)
        loss = criterion(preds, labels)

        val_loss.update(loss.data[0])
        # convert to probabiblity output to cal precision/recall
        preds = F.sigmoid(preds)
        val_score.update(preds.data.cpu().numpy(), val_data[1].numpy())

        if i % opt.log_step == 0:
            logger.info(
                'Epoch [{0}][{1}/{2}]\t'
                'Loss {3:0.7f}\t'.format(
                    0, i, len(val_loader), loss.data[0]))

    return val_loss, val_score


def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every [lr_update] epochs"""
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


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
    .. Check with the average_precision_score method in the sklearn.metrics package
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

    def map(self):
        return 0 if self.num_samples == 0 else self.sum_ap / self.num_samples

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
