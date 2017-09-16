import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from utils import AverageMeter, AverageScore, str2bool

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
                'Loss {3:0.5f}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(
                    epoch, i, len(train_loader), loss.data[0], 
                    batch_time=batch_time,
                    data_time=data_time))
        
        end = time.time()

def validate(opt, model, criterion, val_loader):
    # compute the encoding for all the validation images and captions
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
                'Loss {3:0.5f}\t'.format(
                    0, i, len(val_loader), loss.data[0]))
    
    logger.info('Val score: \n%s', val_score)
    return val_loss.avg

def save_checkpoint(state, filename):
    torch.save(state, filename)
        
def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('train_label', type=str, help='path to the h5file containing the labels')
    parser.add_argument('val_label', type=str, help='path to the h5file containing the labels')
    parser.add_argument('train_imageinfo', type=str, help='imageinfo contains image path')
    parser.add_argument('val_imageinfo', type=str, help='imageinfo contains image path')
    parser.add_argument('output_file', type=str, help='output model file')
    
    parser.add_argument('--train_image_dir', type=str, help='image dir')
    parser.add_argument('--val_image_dir', type=str, help='image dir')
    
    # Optimization: General
    parser.add_argument('--max_patience', type=int, default=5, help='max number of epoch to run since the minima is detected -- early stopping')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size (there will be x seq_per_img sentences)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lr_update', default=10, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--val_step', type=int, default=1, help='val step, default=1 (every epoch)')
    
    # Model settings
    parser.add_argument('--finetune', type=str2bool, default=False,
                        help='Fine-tune the image encoder.')
    parser.add_argument('--cnn_type', default='vgg19', choices=['vgg19', 'resnet152'],
                        help="""The CNN used for image encoder
                        (e.g. vgg19, resnet152)""")
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers')
    parser.add_argument('--num_epochs', type=int, default=30, help='max number of epochs to run for (-1 = run forever)')
    parser.add_argument('--grad_clip', type=float, default=0.1, help='clip gradients at this value (note should be lower than usual 5 because we normalize grads by both batch and seq_length)')
    
    # Evaluation/Checkpointing
    
    parser.add_argument('--log_step', type=int, default=20, help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
    parser.add_argument('--loglevel', type=str, default='DEBUG', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    
    ## misc
    parser.add_argument('--seed', type=int, default=123, help='random number generator seed to use')
    
    opt = parser.parse_args()

    logging.basicConfig(level=getattr(logging, opt.loglevel.upper()),
                        format='%(asctime)s:%(levelname)s: %(message)s')
    
    logger.info('Input arguments: %s', json.dumps(vars(opt), sort_keys=True, indent=4))
    
    # Set the random seed manually for reproducibility.
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.seed)
    else:
        torch.manual_seed(opt.seed)
    
    train_opt = {'label_file': opt.train_label, 
        'imageinfo_file': opt.train_imageinfo,         
        'image_dir': opt.train_image_dir,
        'batch_size': opt.batch_size,         
        'num_workers': opt.num_workers,
        'train': True
    }
    
    val_opt = {'label_file': opt.val_label, 
        'imageinfo_file': opt.val_imageinfo,
        'image_dir': opt.val_image_dir,
        'batch_size': opt.batch_size,       
        'num_workers': opt.num_workers,       
        'train': False
    }
                
    train_loader = get_data_loader(train_opt)
    val_loader = get_data_loader(val_opt)
    
    logger.info('Building model...')
    
    num_labels = train_loader.dataset.get_num_labels()
    
    model = EncoderImageFull(num_labels, finetune=opt.finetune, cnn_type=opt.cnn_type)
    criterion = nn.MultiLabelSoftMarginLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    
    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()
    
    logger.info("Start/continue training...")
    best_loss = sys.maxint
    best_epoch = 0
    
    for epoch in range(opt.num_epochs):
        learning_rate = adjust_learning_rate(opt, optimizer, epoch)
        logger.info('===> Learning rate: %f: ', learning_rate)

        # train for one epoch
        train(opt, model, criterion, optimizer, train_loader, epoch)

        # validate at every val_step epoch
        if epoch % opt.val_step == 0:
            logger.info("Start evaluating...")
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
