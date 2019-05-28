import os
import torch
import torch.optim as optim


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def get_optimizer(cfg, model, state_dict=None):
    optimizer = None
    if cfg.optim == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=0.9,
            weight_decay=1e-5,
            nesterov=True
        )
    elif cfg.optim == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.lr
        )
    elif cfg.optim == 'rmsprop':
        pass
    if state_dict:
        optimizer.load_state_dict(state_dict)

    return optimizer


# from imagenet example of pytorch: https://github.com/pytorch/examples/blob/master/imagenet/main.py
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(optimizer, epoch, args, steps=(20, 40), dec_rate=0.1):
    """Decreases the learning rate to the initial LR decayed by dec_rate every given step """
    assert type(steps) in [list, tuple, int]
    changed_flag = False
    lr = optimizer.param_groups[0]['lr']
    if type(steps) == int:
        if epoch % steps == 0:
            changed_flag = True
    else:
        if epoch in steps:
            changed_flag = True
    if changed_flag:
        lr *= dec_rate
        for param_group in optimizer.param_groups:
            print('Decreasing learning rate to {:1.5f}'.format(lr))
            param_group['lr'] = lr
    # step = 20
    # if args.lr * (0.1 ** (epoch // step)) > 1e-5:
    #     lr = args.lr * (0.1 ** (epoch // step))
    #     for param_group in optimizer.param_groups:
    #         if param_group['lr'] != lr:
    #             print('Decreasing learning rate to {:1.5f}'.format(lr))
    #         param_group['lr'] = lr


def save_checkpoint(states, output_dir, is_best=False, filename='checkpoint.pth'):
    if is_best:
        torch.save(states, filename + '_model_best.pth')
    else:
        torch.save(states, filename + '.pth')
