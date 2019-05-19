import os
import torch
from .resnet import resnet
from .inceptionresnetv2 import inceptionresnetv2
from .inception import inception_v3
from .inceptionv4 import inceptionv4


def generate_model(arch, dataset_numclasses, state_dict=None):
    if 'inceptionresnet' in arch:
        model = eval('inceptionresnetv2')(dataset_numclasses, state_dict)
    if 'inceptionv4' in arch:
        model = eval('inceptionv4')(dataset_numclasses, state_dict)
    elif 'resnet' in arch:
        model = eval('resnet')(arch, dataset_numclasses, state_dict)
    elif 'inception' == arch:
        model = eval('inception_v3')(dataset_numclasses, state_dict)

    model.cuda()
    return model



