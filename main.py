import os
import torch
import torch.nn as nn
from dataloaders import  dataloader_factory as data_fact
from networks import model_factory
from config import parse_args
from utils.utils import get_optimizer, save_checkpoint, adjust_learning_rate
from utils.functions import train_epoch, validate_epoch, test_cassava


def main():

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

    args = parse_args()
    print("Model input size: ", args.model_input_size)
    print("Batch size: ", args.batch_size)
    print('Arch: ', args.arch)
    print('Optimizer: ', args.optim)
    print('Weighted loss: ', args.use_weighted_loss)

    dir_path = os.path.dirname(__file__)

    print('Loading dataset and dataloader..')
    train_percentage = args.train_percentage
    if not args.validate:
        train_percentage = 1
    train_set, train_loader = data_fact.get_dataloader(args, 'train', train_percentage=train_percentage)
    # if args.subset_finetune:
    #     train_set, train_loader = data_fact.get_dataloader(args, 'subset')
    if args.validate:
        val_set, val_loader = data_fact.get_dataloader(args, 'val', train_percentage=train_percentage)
    if args.test:
        test_set, test_loader = data_fact.get_dataloader(args, 'test')
    num_classes = len(train_set.classes)

    class_weights = [1] * num_classes
    if args.use_weighted_loss:
        class_weights = train_set.class_weights
        print("Class weights: ", class_weights)
    class_weights = torch.Tensor(class_weights)

    criterion = nn.CrossEntropyLoss(class_weights)
    if args.use_cuda:
        criterion = criterion.cuda()

    best_perf1 = 0
    best_perf5 = 0
    begin_epoch = 0
    best_epoch = 0
    state_dict = None
    optimizer_dict = None

    if args.resume_path:
        print('Loading finetuned model from {}..', args.resume_path)
        checkpoint = torch.load(args.resume_path)
        begin_epoch = checkpoint['epoch']
        # best_epoch = begin_epoch
        best_epoch = checkpoint['best_epoch']
        best_perf1 = checkpoint['perf1']
        best_perf5 = checkpoint['perf5']
        args.arch = checkpoint['arch']
        num_classes = checkpoint['num_classes']
        state_dict = checkpoint['state_dict']
        optimizer_dict = checkpoint['optimizer']
        print('Begin epoch: ', begin_epoch)
        print('Best Acc@1 at epoch {}: {}'.format(best_epoch, best_perf1))
        # scheduler.load_state_dict(checkpoint['scheduler'])

    model = model_factory.generate_model(args.arch, num_classes, state_dict, args.use_cuda)
    optimizer = get_optimizer(args, model, optimizer_dict)
    if args.subset_finetune:
        args.lr = 1e-6  # do balanced subset finetuning with a smaller learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
            begin_epoch = 0
    print('Learning rate: {:1.5f}', optimizer.param_groups[0]['lr'])

    if args.train:
        for epoch in range(begin_epoch, args.num_epochs):
            print('Epoch: {} / {}'.format(epoch+1, args.num_epochs))

            perf_indicator1, perf_indicator5 = train_epoch(
                epoch, train_loader, model, criterion, optimizer, args.use_cuda)

            if args.validate:
                perf_indicator1, perf_indicator5 = validate_epoch(
                    val_loader, model, criterion, args.use_cuda)

            if perf_indicator1 >= best_perf1:
                best_perf1 = perf_indicator1
                best_perf5 = perf_indicator5
                best_epoch = epoch

                checkpoint_file = '{}_{}_{}_{}{}{}'.format(
                    args.model_input_size, args.arch, args.optim,
                    args.batch_size, '_subset' if args.subset_finetune else '',
                    '_weightedloss' if args.use_weighted_loss else '')

                save_checkpoint({
                    'epoch': epoch + 1,
                    'best_epoch': best_epoch + 1,
                    'perf1': best_perf1,
                    'perf5': best_perf5,
                    'arch': args.arch,
                    'num_classes': model.num_classes,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    # 'scheduler': scheduler.state_dict(),
                }, dir_path, is_best=True, filename=checkpoint_file)

            if (epoch+1) % 5 == 0:
                checkpoint_file = 'Epoch{}_{}_{}_{}_{}{}{}'.format(
                    epoch + 1, args.model_input_size, args.arch, args.optim,
                    args.batch_size, '_subset' if args.subset_finetune else '',
                    '_weightedloss' if args.use_weighted_loss else '')

                save_checkpoint({
                    'epoch': epoch + 1,
                    'best_epoch': best_epoch + 1,
                    'perf1': best_perf1,
                    'perf5': best_perf5,
                    'arch': args.arch,
                    'num_classes': model.num_classes,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    # 'scheduler': scheduler.state_dict(),
                }, dir_path, filename=checkpoint_file)

            print('Epoch {} perf acc@1: {}, perf acc@5: {}'.format(
                epoch+1, perf_indicator1, perf_indicator5))
            print('Best perf acc@1: {}, perf acc@5: {} at epoch {}'.format(
                best_perf1, best_perf5, best_epoch+1))
            # scheduler.step(perf_indicator1)
            if epoch+1 < 100:
                adjust_learning_rate(optimizer, epoch+1, args, steps=2, dec_rate=0.9)

    if args.test:
        test_cassava(test_loader, model, train_set.classes, args.tencrop_test, args)


if __name__ == '__main__':
    main()

