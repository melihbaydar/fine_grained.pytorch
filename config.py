import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Main argument parser')
    parser.add_argument("--use_cuda", type=str2bool,
                        nargs='?', default=True,
                        help="GPU flag.")
    parser.add_argument('--model_input_size',
                        help='Input resolution to feed to network',
                        type=int,
                        default=448)
    parser.add_argument('--batch_size',
                        help='Batch size for training',
                        type=int,
                        default=32)
    parser.add_argument('--num_epochs',
                        help='Number of training training',
                        type=int,
                        default=120)
    parser.add_argument('--arch',
                        help='Network architecture',
                        type=str,
                        default='resnet50')
    parser.add_argument('--optim',
                        help='Optimizer',
                        type=str,
                        default='sgd')
    parser.add_argument('--lr',
                        help='Learning rate',
                        type=float,
                        default=1e-2)
    parser.add_argument('--train_percentage',
                        help='Training set percentage, val set will not be used if set to 1',
                        type=float,
                        default=1)
    parser.add_argument("--train", type=str2bool,
                        nargs='?', default=True,
                        help="Training flag.")
    parser.add_argument("--use_extraimages", type=str2bool,
                        nargs='?', default=False,
                        help="Use labeled extra images in train set if true.  "
                             "P.S. extra images should be labeled beforehand.")
    parser.add_argument("--validate", type=str2bool,
                        nargs='?', default=False,
                        help="Validation flag.")
    parser.add_argument("--test", type=str2bool,
                        nargs='?', default=True,
                        help="Test flag.")
    parser.add_argument("--subset_finetune", type=str2bool,
                        nargs='?', default=False,
                        help="Balanced subset finetuning flag.")
    parser.add_argument("--use_weighted_loss", type=str2bool,
                        nargs='?', default=False,
                        help="Use class weights in loss function against imbalanced data if true.")
    parser.add_argument("--tencrop_test", type=str2bool,
                        nargs='?', default=False,
                        help="Tencrop testing mode flag.")
    parser.add_argument("--resume_path", type=str,
                        default='',
                        help="Resume training from checkpoint.")
    args = parser.parse_args()

    return args

