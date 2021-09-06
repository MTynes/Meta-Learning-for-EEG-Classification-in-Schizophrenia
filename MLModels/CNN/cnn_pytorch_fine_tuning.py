import os
import shutil
import sys
import argparse
import time
import itertools
import platform

import numpy as np
import pandas as pd
import json
from PIL import Image
import torch
import torch.nn as nn
import warnings
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import scikitplot as skplt
from torch.autograd import Variable
from torch.backends import cudnn
from torch.nn import DataParallel
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# File modified from main.py and /utils.util.py
# of this repo https://github.com/zhangrong1722/CheXNet-Pytorch

plt.switch_backend('agg')

parser = argparse.ArgumentParser(description='Training on EEG Spectrograms')

parser.add_argument('--train_dir', type=str, help='train data directory',
                    default='/content/miniimagenet/images')
parser.add_argument('--fine_tuning_dir', type=str, help='fine tuning data directory',
                    default='/content/all_fine_tuning/images')
parser.add_argument('--validation_dir', type=str, help='validation data directory',
                    default='/content/all_validation_images')
parser.add_argument('--test_dir', type=str, help='test data directory',
                    default='/content/all_test_images')
parser.add_argument('--imgsz', type=int, help='image size to be used as both width and height', default=120)  #
parser.add_argument('--plot_cm', default=False, type=bool, help='Boolean to calculate and plot confusion matrix')
parser.add_argument('--plot_roc', default=False, type=bool, help='Boolean to plot Receiver Operating Characteristic (ROC)')
parser.add_argument('--fine_tuning_epochs', '-fte', default=200, type=int, help='Fine tuning epochs')

parser.add_argument('--batch_size', '-b', default=10, type=int, help='batch size')
parser.add_argument('--epochs', '-e', default=90, type=int, help='training epochs')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='use gpu or not')
parser.add_argument('--step_size', default=30, type=int, help='learning rate decay interval')
parser.add_argument('--gamma', default=0.1, type=float, help='learning rate decay scope')
parser.add_argument('--interval_freq', '-i', default=12, type=int, help='printing log frequency')
parser.add_argument('--prefix', '-p', default='classifier', type=str, help='folder prefix')
parser.add_argument('--best_model_path', default='model_best.pth.tar', help='best model saved path')

best_acc = 0.0


def main():
    global args, best_acc
    args = parser.parse_args()
    # save source script
    set_prefix(args.prefix, __file__)
    model = CNN() #models.densenet121(pretrained=False, num_classes=2)
    if args.cuda:
        model = DataParallel(model).cuda()
    else:
        warnings.warn('No GPU detected.')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # accelerate the speed of training
    cudnn.benchmark = True

    base_train_loader, fine_tuning_loader, val_loader, test_loader = load_dataset()
    class_names = base_train_loader.dataset.classes
    print('Class names: ', ', '.join(class_names))
    criterion = nn.CrossEntropyLoss().cuda()
    metrics_by_epoch = []  # store metrics

    # learning rate decay per epochs
    # TODO check if scheduler should be within the first for loop
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    since = time.time()
    print('-' * 10)
    # loop over training commands using basic train arguments and fine tuning arguments
    epoch_params = [args.epochs, args.fine_tuning_epochs]
    train_loaders = [base_train_loader, fine_tuning_loader]
    training_modes = ['main training cycle', 'fine tuning cycle']
    for i in range(2):
        for epoch in range(epoch_params[i]):
            exp_lr_scheduler.step()
            train_loss, train_acc = train(train_loaders[i], model, optimizer, criterion, epoch, training_modes[i])
            cur_loss, cur_accuracy = validate(model, val_loader, criterion)
            metrics_by_epoch.append({'train_loss': train_loss, 'train_acc': train_acc * 100, \
                                     'val_loss': cur_loss, 'val_acc': float(cur_accuracy.item())})
            is_best = cur_accuracy > best_acc
            best_acc = max(cur_accuracy, best_acc)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'resnet18',
                'state_dict': model.state_dict(),
                'best_accuracy': best_acc,
                'optimizer': optimizer.state_dict(),
                'train_type': training_modes[i]
            }, is_best)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # compute validate meter such as confusion matrix
    compute_validate_meter(model, add_prefix(args.prefix, args.best_model_path), val_loader)
    # save running parameter setting to json
    write(vars(args), add_prefix(args.prefix, 'paras.txt'))
    test(model, test_loader, criterion)
    pd.DataFrame(metrics_by_epoch).to_csv('metrics_by_epoch.csv')

# adapted from https://www.pluralsight.com/guides/image-classification-with-pytorch
# using default stride of 1 and default padding of 0 for conv2d layers
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        #self.conv3 = nn.Conv2d(32, 32, kernel_size=3)
        #self.conv2_drop = nn.Dropout2d()
        self.batch_norm = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(6272, 800)
        self.fc2 = nn.Linear(800, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.batch_norm(x)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.batch_norm(x)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.batch_norm(x)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.batch_norm(x)
        x = x.view(x.shape[0],-1)
        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


def compute_validate_meter(model, best_model_path, val_loader):
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['state_dict'])
    best_acc = checkpoint['best_accuracy']
    print('\nBest validation accuracy={:.4f}%'.format(best_acc))
    pred_y = list()
    test_y = list()
    probas_y = list()
    for data, target in val_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
            output = model(data)
            probas_y.extend(output.data.cpu().numpy().tolist())
            pred_y.extend(output.data.cpu().max(1, keepdim=True)[1].numpy().flatten().tolist())
            test_y.extend(target.data.cpu().numpy().flatten().tolist())

    if args.plot_cm:
        confusion = confusion_matrix(pred_y, test_y)
        plot_confusion_matrix(confusion,
                          classes=val_loader.dataset.classes,
                          title='Confusion matrix')
    if args.plot_roc:
        plt_roc(test_y, probas_y)
    pd.DataFrame({'pred_y': pred_y, 'probas_y': probas_y, 'test_y': test_y}).to_csv('best_val_model_predictions.csv')


###### utility functions
def add_prefix(prefix, path):
    return os.path.join(prefix, path)


def set_prefix(prefix, name):
    if not os.path.isdir(prefix):
        os.mkdir(prefix)
    if platform.system() == 'Windows':
        name = name.split('\\')[-1]
    else:
        name = name.split('/')[-1]
    shutil.copy(name, os.path.join(prefix, name))


def write(dic, path):
    with open(path, 'w+') as f:
        f.write(json.dumps(dic))


########


def plt_roc(test_y, probas_y, plot_micro=False, plot_macro=False):
    assert isinstance(test_y, list) and isinstance(probas_y, list), 'the type of input must be list'
    skplt.metrics.plot_roc(test_y, probas_y, plot_micro=plot_micro, plot_macro=plot_macro)
    plt.savefig(add_prefix(args.prefix, 'roc_auc_curve.png'))
    plt.close()


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    refence:
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(add_prefix(args.prefix, 'confusion_matrix.png'))
    plt.close()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    # save training state after each epoch
    torch.save(state, add_prefix(args.prefix, filename))
    if is_best:
        shutil.copyfile(add_prefix(args.prefix, filename),
                        add_prefix(args.prefix, args.best_model_path))


def load_dataset():
    mean = [0.5186, 0.5186, 0.5186]
    std = [0.1968, 0.1968, 0.1968]
    normalize = transforms.Normalize(mean, std)
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.NEAREST),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_test_transforms = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.NEAREST),
        transforms.ToTensor(),
        normalize,
    ])


    base_train_dataset = ImageFolder(root=args.train_dir, transform=train_transforms)
    fine_tuning_dataset = ImageFolder(root=args.fine_tuning_dir, transform=train_transforms)
    validation_dataset = ImageFolder(root=args.validation_dir, transform=val_test_transforms)
    test_dataset = ImageFolder(root=args.test_dir, transform=val_test_transforms)

    train_data_loader = DataLoader(base_train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=4,
                                   pin_memory=True if args.cuda else False)
    fine_tuning_data_loader = DataLoader(fine_tuning_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=4,
                                   pin_memory=True if args.cuda else False)
    validation_data_loader = DataLoader(validation_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        num_workers=4,
                                        pin_memory=True if args.cuda else False)
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=True if args.cuda else False)

    print('Data loading complete.')
    print('Train directory: ', args.train_dir)
    print('Fine tuning directory: ', args.fine_tuning_dir)
    print('Validation directory: ', args.test_dir)
    print('Test directory: ', args.test_dir)
    print('Batch size: ', args.batch_size)
    print('Length of train loader: ', len(train_data_loader))
    print('Length of validation loader: ', len(validation_data_loader))
    print('Length of test loader: ', len(test_data_loader))
    return train_data_loader, fine_tuning_data_loader, validation_data_loader, test_data_loader


def train(train_loader, model, optimizer, criterion, epoch, training_mode):
    model.train(True)
    print('Mode: {}   Epoch {}/{}'.format(training_mode, epoch + 1, args.epochs))
    print('-' * 10)
    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for idx, (inputs, labels) in enumerate(train_loader):
        # wrap them in Variable
        if args.cuda:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)

        _, preds = torch.max(outputs.data, 1)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if idx % args.interval_freq == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, idx * len(inputs), len(train_loader.dataset),
                100. * idx / len(train_loader), loss.data.item()))  # [0]))

        # statistics
        running_loss += loss.data.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = float(running_loss) / len(train_loader.dataset)
    epoch_acc = float(running_corrects) / len(train_loader.dataset)

    print('Training Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc * 100.))
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    print('length of validation loader: ', len(val_loader))
    for data, target in val_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
            output = model(data)
            val_loss += criterion(output, target).data.item()  # [0]
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    val_loss /= len(val_loader.dataset)
    val_acc = 100. * correct / len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset), val_acc))
    return val_loss, val_acc


def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    print('length of test loader: ', len(test_loader))
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += criterion(output, target).data.item()  # [0]
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_acc))
    return test_acc


if __name__ == '__main__':
    main()
