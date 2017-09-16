import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import os
import sys
import time
import math
import json

import logging
from datetime import datetime
from model import EncoderImageFull
from dataloader import get_data_loader

logger = logging.getLogger(__name__)
from utils import AverageMeter

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
        end = time.time()
        
        # Print log info
        if i % opt.log_step == 0:
            logger.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{3}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(
                    epoch, i, len(train_loader), loss.data[0], 
                    batch_time=batch_time,
                    data_time=data_time))


def validate(opt, model, criterion, val_loader):
    # compute the encoding for all the validation images and captions
    val_loss = AverageMeter()
    
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
        
        if i % opt.log_step == 0:
            logger.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{3}\t'.format(
                    0, i, len(val_loader), loss.data[0]))
    
    return val_loss.avg.cpu()

def save_checkpoint(state, filename):
    torch.save(state, filename)

def load_checkpoint(filename='model_best.pth.tar', prefix=''):
    checkpoint_file = os.path.join(prefix, filename)
    logger.info('Loading checkpoint: %s', checkpoint_file)
    if os.path.isfile(checkpoint_file):
        return torch.load(checkpoint_file)
    return None
        
def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = opt.learning_rate * (0.1 ** (epoch // opt.learning_rate))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('output_file', type=str, help='output model file')
    parser.add_argument('--train_label_file', type=str, help='path to the h5file containing the preprocessed dataset')
    parser.add_argument('--val_label_file', type=str, help='path to the h5file containing the preprocessed dataset')
    parser.add_argument('--test_label_file', type=str, help='path to the h5file containing the preprocessed dataset')

    parser.add_argument('--train_imageinfo_file', type=str, help='Gold captions in MSCOCO format to cal language metrics')
    parser.add_argument('--val_imageinfo_file', type=str, help='Gold captions in MSCOCO format to cal language metrics')
    parser.add_argument('--test_imageinfo_file', type=str, help='Gold captions in MSCOCO format to cal language metrics')
    
    parser.add_argument('--train_image_dir', type=str, help='image dir')
    parser.add_argument('--val_image_dir', type=str, help='image dir')
    parser.add_argument('--test_image_dir', type=str, help='image dir')
    
    # Optimization: General
    parser.add_argument('--max_patience', type=int, default=50, help='max number of epoch to run since the minima is detected -- early stopping')
    parser.add_argument('--batch_size', type=int, default=128, help='Video batch size (there will be x seq_per_img sentences)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--val_step', type=int, default=10, help='learning rate')
    
    # Model settings
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--num_epochs', type=int, default=30, help='max number of epochs to run for (-1 = run forever)')
    parser.add_argument('--grad_clip', type=float, default=0.1, help='clip gradients at this value (note should be lower than usual 5 because we normalize grads by both batch and seq_length)')
    
    # Evaluation/Checkpointing
    parser.add_argument('--save_checkpoint_from', type=int, default=20, help='Start saving checkpoint from this epoch')
    parser.add_argument('--save_checkpoint_every', type=int, default=1, help='how often to save a model checkpoint in epochs?')
    
    parser.add_argument('--checkpoint_path', type=str, default='output/model', help='folder to save checkpoints into (empty = this folder)')
    parser.add_argument('--log_step', type=int, default=20, help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
    parser.add_argument('--loglevel', type=str, default='DEBUG', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    
    ## misc
    parser.add_argument('--backend', type=str, default='cudnn', help='nn|cudnn')
    parser.add_argument('--id', type=str, default=None, help='an id identifying this run/job. used in cross-val and appended when writing progress files')
    parser.add_argument('--seed', type=int, default=123, help='random number generator seed to use')
    
    parser.add_argument('--test_checkpoint',  type=str, default='', help='path to the checkpoint needed to be tested')
    parser.add_argument('--test_only', type=int, default=0, help='1: use the current model (located in current path) for testing')
    
    opt = parser.parse_args()

    logging.basicConfig(level=getattr(logging, opt.loglevel.upper()),
                        format='%(asctime)s:%(levelname)s: %(message)s')
    
    logger.info('Input arguments: %s', json.dumps(vars(opt), sort_keys=True, indent=4))
    
    # Set the random seed manually for reproducibility.
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.seed)
    else:
        torch.manual_seed(opt.seed)
    
    train_opt = {'label_file': opt.train_label_file, 
        'batch_size': opt.batch_size,
        'imageinfo_file': opt.train_imageinfo_file,
        'image_dir': opt.train_image_dir,
        'num_workers': opt.num_workers,
        'train': True
    }
    
    val_opt = {'label_file': opt.val_label_file, 
        'batch_size': opt.batch_size,
        'imageinfo_file': opt.val_imageinfo_file,
        'image_dir': opt.val_image_dir,
        'num_workers': opt.num_workers,       
        'train': False
    }
    
    test_opt = {'label_file': opt.test_label_file, 
        'batch_size': opt.batch_size,
        'imageinfo_file': opt.test_imageinfo_file,
        'image_dir': opt.test_image_dir,        
        'num_workers': opt.num_workers,        
        'train': False
    }
                
    train_loader = get_data_loader(train_opt)
    val_loader = get_data_loader(val_opt)
    test_loader = get_data_loader(test_opt)
    
    logger.info('Building model...')
    
    num_labels = train_loader.dataset.get_num_labels()
    
    model = EncoderImageFull(num_labels)
    criterion = nn.MultiLabelSoftMarginLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    
    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()
    
    logger.info("=> start/continue training...")
    best_loss = sys.maxint
    best_epoch = 0
    
    for epoch in range(opt.num_epochs):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        train(opt, model, criterion, optimizer, train_loader, epoch)

        # validate at every val_step
        if epoch % opt.val_step == 0:
            loss = validate(opt, model, criterion, val_loader)

            if loss < best_loss:
                logger.info('Found new best score: %.4f, previous score: %.4f', loss, best_loss)
                best_loss = loss
                best_epoch = epoch
                
                logger.info('Saving new checkpoint to: %s', opt.output_file)    
                save_checkpoint({
                        'epoch': epoch + 1,
                        'model': model.state_dict(),
                        'best_loss': best_loss,
                        'best_epoch': best_epoch,    
                        'opt': opt
                    }, opt.output_file)
                
            else:
                logger.info('Current score: %.4f, best score is %.4f @ epoch %d', loss, best_loss, best_epoch)
        
        if epoch - best_epoch > opt.max_patience:
            logger.info('Terminated by early stopping!')
            break;    
