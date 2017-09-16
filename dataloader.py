import os
import sys
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
import numpy as np
import h5py
import json
from PIL import Image

assert 'CLCV_HOME' in os.environ, 'CLCV_HOME is not set!'

sys.path.append(os.path.join(os.environ['CLCV_HOME'], 'utils'))
from csrmatrix import load_csrmatrix

import logging
from datetime import datetime
logger = logging.getLogger(__name__)

def get_image_transform(train=True, scale_size=256, crop_size=224):
    
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    t_list = []
    if train:
        t_list = [transforms.RandomSizedCrop(crop_size),
                  transforms.RandomHorizontalFlip()]
    else:
        t_list = [transforms.Scale(scale_size), transforms.CenterCrop(crop_size)]

    t_end = [transforms.ToTensor(), normalizer]
    transform = transforms.Compose(t_list + t_end)
    return transform

class DataLoader(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f8k, f30k, coco, 10crop
    """

    def __init__(self, opt):
        
        self.image_dir = opt['image_dir']
        self.image_transform = get_image_transform(train = opt['train'])
        
        logger.info('Loading label file: %s', opt['label_file'])
        with h5py.File(opt['label_file'], 'r') as f:
            self.vocab = [i for i in f['vocab']]
            self.image_ids = [i for i in f['index']]
            self.label_data = load_csrmatrix(f)
            self.num_labels = np.array(f['data']['shape'])[1]
        
        logger.info('Loading imageinfo file: %s', opt['imageinfo_file'])
        imageinfo = json.load(open(opt['imageinfo_file']))
        self.file_names = {_['id']:_['file_name'] for _ in imageinfo['images']}
        
    def __getitem__(self, index):
        
        image_id = self.image_ids[index]
        file_name = self.file_names[image_id]
        file_path = os.path.join(self.image_dir, file_name)
        image = Image.open(file_path).convert('RGB')
        image = self.image_transform(image)
        label = self.label_data[index].toarray().squeeze().astype(np.float32)
        return image, label

    def __len__(self):
        return len(self.image_ids)
    
    def get_vocab(self):
        return self.vocab
    
    def get_vocab_size(self):
        return len(self.vocab)

    def get_num_labels(self):
        return self.num_labels
    
def get_data_loader(opt):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
        
    dataset = DataLoader(opt)
        
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              pin_memory=False,
                                              num_workers=opt['num_workers'],
                                              batch_size=opt['batch_size'],
                                              shuffle=opt['train'])

    return data_loader
