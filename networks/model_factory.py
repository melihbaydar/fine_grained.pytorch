from .resnet import resnet
from .inceptionresnetv2 import inceptionresnetv2
from .inception import inception_v3
from .inceptionv4 import inceptionv4


def generate_model(arch, dataset_numclasses, state_dict=None, use_cuda=True):
    if 'inceptionresnet' in arch:
        model = inceptionresnetv2(dataset_numclasses, state_dict)
    if 'inceptionv4' in arch:
        model = inceptionv4(dataset_numclasses, state_dict)
    elif 'resnet' in arch:
        model = resnet(arch, dataset_numclasses, state_dict)
    elif 'inception' == arch:
        model = inception_v3(dataset_numclasses, state_dict)
    if use_cuda:
        model.cuda()
    return model



