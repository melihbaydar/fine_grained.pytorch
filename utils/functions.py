import os
import time
import torch
import torch.nn.functional as F

from utils import utils
import pandas as pd
import numpy as np
from utils import _init_paths
from sklearn.metrics import confusion_matrix


def train_epoch(epoch, train_loader, model, criterion, optimizer, use_cuda=True):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    progress = utils.ProgressMeter(
        len(train_loader), batch_time, data_time,
        top1, top5, losses, prefix="Epoch: [{}]".format(epoch + 1))

    print_freq = len(train_loader) // 4 + 1
    all_preds = []
    all_labels = []
    model.train()
    end = time.time()
    for i, (paths, inputs, labels) in enumerate(train_loader):

        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        data_time.update(time.time() - end)

        # forward + backward + optimize
        if type(model).__name__ == 'Inception3' and model.aux_logits:
            outputs, aux_outputs = model(inputs)
            loss_aux = criterion(aux_outputs, labels)
            loss_final = criterion(outputs, labels)
            loss = loss_final + 0.4*loss_aux
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        acc1, acc5 = utils.accuracy(outputs, labels, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))

        # for confusion matrix calculation
        _, preds = outputs.topk(1, 1, True, True)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

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

    print(confusion_matrix(all_labels, all_preds))
    return top1.avg, top5.avg


def validate_epoch(val_loader, model, criterion, use_cuda=True):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    progress = utils.ProgressMeter(len(val_loader), batch_time, top1, top5, losses,
                                   prefix='Val: ')

    # switch to evaluate mode
    all_preds = []
    all_labels = []
    model.eval()
    print_freq = len(val_loader) // 4 + 1
    with torch.no_grad():
        end = time.time()
        for i, (_, inputs, labels) in enumerate(val_loader):
            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # measure accuracy and record loss
            acc1, acc5 = utils.accuracy(outputs, labels, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

            # for confusion matrix calculation
            _, preds = outputs.topk(1, 1, True, True)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0 or i+1 == len(val_loader):
                progress.print(i+1)

        print(confusion_matrix(all_labels, all_preds))
        return top1.avg, top5.avg


def test_cassava(test_loader, model, class_names, args):
    end = time.time()
    model.eval()
    preds = []
    image_names = []
    softmax_outs = []
    report_freq = len(test_loader) // 9 + 1
    with torch.no_grad():
        for i, (path, input, label) in enumerate(test_loader):
            # print(path)
            image_names.append(path[0].split('/')[-1])
            if args.use_cuda:
                input = input.cuda()
            if args.tencrop_test:
                bs, ncrops, c, h, w = input.size()
                outputs = model(input.view(-1, c, h, w))  # fuse batch size and ncrop
                outputs = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
            else:
                outputs = model(input)
            softmax_outs.extend(F.softmax(outputs).cpu().numpy())
            _, pred = outputs.topk(1, 1, True, True)
            preds.append(class_names[pred])

            if i % report_freq == 0 or i+1 == len(test_loader):
                print('Processed test data: ', i+1)

    results_dict = {'Category': preds, 'Id': image_names}
    results_df = pd.DataFrame(results_dict)
    submission_file = 'Epoch{}_{}_{}_{}_{}{}{}'.format(
        args.num_epochs, args.model_input_size, args.arch,
        args.optim, args.batch_size, '_subset' if args.subset_finetune else '',
        '_weightedloss' if args.use_weighted_loss else '')
    i = 1
    # if there is another file with same name, find a new name using i variable in filename
    while os.path.exists(str(i) + '_' + submission_file + '.csv'):
        i += 1
    # export prediction in requested kaggle competition format
    results_df.to_csv(str(i) + '_' + submission_file + '.csv', index=False)
    # save softmax outputs for later ensembling
    np.save(str(i) + submission_file, np.asarray(softmax_outs))
