import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable

import logging
logger = logging.getLogger(__name__)


class DepNet(nn.Module):

    def __init__(
            self,
            num_labels,
            finetune=False,
            cnn_type='vgg19',
            pretrained=True):
        """Load a pretrained model and replace top fc layer."""
        super(DepNet, self).__init__()

        self.finetune = finetune
        self.cnn = self.get_cnn(cnn_type, pretrained)

        # For efficient memory usage.
        for param in self.cnn.parameters():
            param.requires_grad = finetune

        # Replace the last fully connected layer of CNN with a new one
        if cnn_type.startswith('vgg'):
            self.fc = nn.Linear(self.cnn.classifier._modules['6'].in_features,
                                num_labels)
            self.cnn.classifier = nn.Sequential(
                *list(self.cnn.classifier.children())[:-1])

        elif cnn_type.startswith('resnet'):
            self.fc = nn.Linear(self.cnn.module.fc.in_features, num_labels)
            self.cnn.module.fc = nn.Sequential()

    def get_cnn(self, cnn_type, pretrained):
        """Load a pretrained CNN and parallelize over GPUs
        """
        logger.info("===> Loading pre-trained model '{}'".format(cnn_type))
        model = models.__dict__[cnn_type](pretrained=pretrained)

        if cnn_type.startswith('alexnet') or cnn_type.startswith('vgg'):
            model.features = nn.DataParallel(model.features)
            model.cuda()
        else:
            model = nn.DataParallel(model).cuda()

        return model

    def parameters(self):
        params = list(self.fc.parameters())
        if self.finetune:
            params += list(self.cnn.parameters())
        return params

    def forward(self, images):
        """Extract image feature vectors."""

        features = self.cnn(images)
        features = self.fc(features)

        return features
