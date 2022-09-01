from args import args
import torch
import torch.nn as nn
import models
import data_gen
from tqdm import tqdm
import torch.utils.data as data
import torchvision.transforms as transforms
from transform import get_transforms
import os
from build_net import make_model
from utils import get_optimizer, AverageMeter, save_checkpoint, accuracy
import torchnet.meter as meter
from sklearn import metrics
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time
import cv2

# Use CUDA
torch.cuda.set_device(1)
use_cuda = torch.cuda.is_available()
# use_cuda = False

num = []
loss = []
acc = []
loss_te =[]
best_recall = 0
best_acc = 0
best_ac = 0
best_auc = 0
best_f1 = 0
best_precision = 0

def val(val_loader, model, criterion, epoch, use_cuda):
    global best_acc
    losses = AverageMeter()
    val_acc = AverageMeter()

    model.eval()  # 将模型设置为验证模式
    # 混淆矩阵
    confusion_matrix = meter.ConfusionMeter(args.num_classes)
    for _, (inputs, targets) in enumerate(val_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # confusion_matrix.add(outputs.data.squeeze(),targets.long())
        acc1 = accuracy(outputs.data, targets.data)

        # measure accuracy and record loss
        losses.update(loss.item(), inputs.size(0))
        val_acc.update(acc1.item(), inputs.size(0))
    return losses.avg, val_acc.avg


def main():
    global best_acc,best_ac,best_auc,best_f1,best_recall,best_precision
    start = time.time()
    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)

    # data
    transformations = get_transforms(input_size=args.image_size, test_size=args.image_size)
    train_set = data_gen.Dataset(root=args.train_txt_path, transform=transformations['val_train'])
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    #val_set = data_gen.ValDataset(root=args.val_txt_path, transform=transformations['val_test'])
    #val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # model
    model = make_model(args)
    if use_cuda:
        model.cuda()

    # define loss function and optimizer
    if use_cuda:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = get_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1, last_epoch=-1)

    # load checkpoint
    start_epoch = args.start_epoch
    now_model_name = 'AMHnet'
    dict = './result/'
    dict = dict + now_model_name
    dict += '.txt'
    with open(dict, 'a') as f:
        # train
        for epoch in range(start_epoch, args.epochs):
            print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))

            train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, use_cuda)

            #test_loss, val_acc = val(val_loader, model, criterion, epoch, use_cuda)

            scheduler.step()

            # print(f'train_loss:{train_loss}\t val_loss:{test_loss}\t train_acc:{train_acc} \t val_acc:{val_acc}')
            print('train_loss:%.4f' % train_loss, 'train_acc:%.4f' % train_acc)

            num.append(epoch + 1)
            loss.append(train_loss)
            acc.append(train_acc)

            # save_model
            is_best = train_acc >= best_acc
            best_acc = max(train_acc, best_acc)

            loss_test,test_acc,test_auc,test_f1,test_recall,test_precision = test(model,now_model_name)
            loss_te.append(loss_test)

            best_ac = max(test_acc, best_ac)
            best_auc = max(test_auc, best_auc)
            best_f1 = max(test_f1, best_f1)
            if(test_recall >best_recall):
                best_precision = max(test_precision, best_precision)
            best_recall = max(test_recall,best_recall)


            word = str(epoch + 1) + ' ' + str(round(train_loss, 4)) + ' ' + str(round(train_acc, 4))+' '+str(round(loss_test, 4))
            f.write(word)
            f.write('\n')
            save_checkpoint({
                'fold': 0,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'train_acc': train_acc,
                'acc': train_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, now_model_name, single=True, checkpoint=args.checkpoint)
        f.write('best_acc is ' + str(best_acc))
    plt.plot(num, loss, color="r", marker="p", linestyle="-", alpha=0.5, mfc="c")  # 添加曲线格式
    plt.savefig(
        './result/AMHNet.png')
    plt.plot(num, loss_te, color="r", marker="p", linestyle="-", alpha=0.5, mfc="c")  # 添加曲线格式
    plt.savefig(
        './result/AMHnet.png')

    print("best acc = ", best_ac)
    print("best auc = ", best_auc)
    print("best F1-score = ", best_f1)
    print("best recall = ", best_recall)
    print("best precision = ", best_precision)
    end = time.time()
    print(str(end - start))


def train(train_loader, model, criterion, optimizer, epoch, use_cuda):
    model.train()
    losses = AverageMeter()
    train_acc = AverageMeter()

    for (inputs, targets) in tqdm(train_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # 梯度参数设为0
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        # measure accuracy and record loss
        acc = accuracy(outputs.data, targets.data)
        # inputs.size(0)=32
        losses.update(loss.item(), inputs.size(0))
        train_acc.update(acc.item(), inputs.size(0))

    return losses.avg, train_acc.avg


def test(model,now_model_name):
    # data
    transformations = get_transforms(input_size=args.image_size, test_size=args.image_size)
    test_set = data_gen.TestDataset(root=args.test_txt_path, transform=transformations['test'])
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    if use_cuda:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()
    losses = AverageMeter()
    '''
    # load model
    model = make_model(args)

    if args.model_path:
        # 加载模型
        model.load_state_dict(torch.load(args.model_path))

    if use_cuda:
        model.cuda()
    '''
    # evaluate
    dict = './resultW/'
    dict = dict + now_model_name
    dict += '_test.txt'
    y_pred = []
    y_true = []
    img_paths = []
    with torch.no_grad():
        model.eval()  # 设置成eval模式
        for (inputs, targets, paths) in tqdm(test_loader):
            y_true.extend(targets.detach().tolist())
            img_paths.extend(list(paths))
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            outputs = model(inputs)  # (16,2)
            loss = criterion(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            probability = torch.max(outputs, dim=1)[1].data.cpu().numpy().squeeze()
            y_pred.extend(probability)

        with open(dict, 'a') as f:
            accuracy = metrics.accuracy_score(y_true, y_pred)
            print("accuracy=", accuracy)

            recall = metrics.recall_score(y_true, y_pred,average='binary')
            print("recall=",recall)
            precision = metrics.precision_score(y_true, y_pred,average="binary", pos_label=1)
            print("precision score=",precision)
            roc_auc =  metrics.roc_auc_score(y_true, y_pred)
            print("roc-auc score=",roc_auc)
            F1_score =  metrics.f1_score(y_true,y_pred)
            print("F1_score=", F1_score)
            f.write('accuracy roc-auc score F1_score')
            f.write('\n')
            word = str(round(accuracy, 4)) +' '+str(round(roc_auc, 4))+' '+str(round(F1_score, 4))+' '+str(round(recall, 4))+' '+str(round(precision, 4))
            f.write(word)
            f.write('\n')
        res_dict = {
            'img_path': img_paths,
            'label': y_true,
            'predict': y_pred,

        }
        '''
        df = pd.DataFrame(res_dict)
        df.to_csv(args.result_csv, index=False)
        print(f"write to {args.result_csv} succeed ")
        for i in range(len(img_paths)):

            if y_pred[i] != y_true[i]:
                img_path = os.path.join(args.dataset_path, img_paths[i])
                img_path = img_path + '.jpg'

                img = cv2.imread(img_path)  # 读取图片
                path = os.path.join('./photo_wrong/', img_paths[i])
                path = path + '.jpg'
                cv2.imwrite(path, img)
        '''
    return losses.avg,accuracy,roc_auc,F1_score,recall,precision

if __name__ == "__main__":
    # main()
    # 划分数据集
    # data_gen.Split_datatset(args.dataset_txt_path,args.train_txt_path,args.test_txt_path)
    # data_gen.Split_datatset(args.train_txt_path,args.train_txt_path,args.val_txt_path)
    if args.mode == 'train':
        main()
    else:
        test(use_cuda)