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
from train import validate
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('test_label', type=str, help='path to the h5file containing the labels')
    parser.add_argument('test_imageinfo', type=str, help='imageinfo contains image path')
    parser.add_argument('model_file', type=str, help='model file')
    parser.add_argument('output_file', type=str, help='output file')
    parser.add_argument('--test_image_dir', type=str, help='image dir')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size (there will be x seq_per_img sentences)')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers')
    parser.add_argument('--loglevel', type=str, default='DEBUG', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    
    opt = parser.parse_args()

    logging.basicConfig(level=getattr(logging, opt.loglevel.upper()),
                        format='%(asctime)s:%(levelname)s: %(message)s')
    
    logger.info('Input arguments: %s', json.dumps(vars(opt), sort_keys=True, indent=4))
    
    if not os.path.isfile(opt.model_file):
        logger.info('Model file does not exist: %s', opt.model_file)
        
    else:
        logger.info('Loading model: %s', opt.model_file)
        test_opt = {
            'label_file': opt.test_label, 
            'imageinfo_file': opt.test_imageinfo,
            'image_dir': opt.test_image_dir,
            'batch_size': opt.batch_size,       
            'num_workers': opt.num_workers,       
            'train': False
        }

        checkpoint = torch.load(opt.model_file)
        
        test_loader = get_data_loader(test_opt)
        num_labels = test_loader.dataset.get_num_labels()
        
        logger.info('Building model...')
        opt = checkpoint['opt']
        model = EncoderImageFull(num_labels, finetune=opt.finetune, cnn_type=opt.cnn_type)
        criterion = nn.MultiLabelSoftMarginLoss()
        model.load_state_dict(checkpoint['model'])
        
        if torch.cuda.is_available():
            model.cuda()
            criterion.cuda()

        logger.info('Start testing...')
        validate(opt, model, criterion, test_loader)
        logger.info('Done')
