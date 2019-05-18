import os
import time
import torch

from utils import utils
import pandas as pd
from utils import _init_paths


def train_epoch(epoch, train_loader, model, criterion, optimizer):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    progress = utils.ProgressMeter(
        len(train_loader), batch_time, data_time, losses,
        top1, top5, prefix="Epoch: [{}]".format(epoch + 1))

    print_freq = len(train_loader) // 4 + 1
    model.train()
    end = time.time()
    for i, (inputs, labels) in enumerate(train_loader):
        # get the inputs
        (_, inputs) = inputs  # remove image paths from inputs tuple
        inputs, labels = inputs.cuda(), labels.cuda()
        data_time.update(time.time() - end)

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        acc1, acc5 = utils.accuracy(outputs, labels, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))

        # zero the parameter gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print statistics
        if i % print_freq == 0 or i + 1 == len(train_loader):
            progress.print(i+1)

    return top1.avg, top5.avg


def validate_epoch(val_loader, model, criterion):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    progress = utils.ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                                   prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    print_freq = len(val_loader) // 4 + 1
    with torch.no_grad():
        end = time.time()
        for i, (inputs, labels) in enumerate(val_loader):
            (_, inputs) = inputs  # remove image paths from inputs tuple
            inputs, labels = inputs.cuda(), labels.cuda()

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # measure accuracy and record loss
            acc1, acc5 = utils.accuracy(outputs, labels, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0 or i+1 == len(val_loader):
                progress.print(i+1)

        return top1.avg, top5.avg


def test_cassava(test_loader, model, class_names, tencrop_test):
    end = time.time()
    model.eval()
    preds = []
    image_names = []
    report_freq = len(test_loader) // 9 + 1
    with torch.no_grad():
        for i, (input, label) in enumerate(test_loader):
            (path, input) = input  # remove image paths from inputs tuple
            # print(path)
            image_names.append(path[0].split('/')[-1])
            input = input.cuda()
            if tencrop_test:
                bs, ncrops, c, h, w = input.size()
                outputs = model(input.view(-1, c, h, w))  # fuse batch size and ncrop
                outputs = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
            else:
                outputs = model(input)
            _, pred = outputs.topk(1, 1, True, True)
            preds.append(class_names[pred])

            if i % report_freq == 0 or i+1 == len(test_loader):
                print('Processed test data: ', i+1)

    results_dict = {'Category': preds, 'Id': image_names}
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv('Submission.csv', index=False)
