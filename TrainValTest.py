from __future__ import division

import random
random.seed(0)
import pickle
import math
import os
import sys
import numpy as np
import torch
torch.manual_seed(1)
# torch.use_deterministic_algorithms(True)
import time
np.set_printoptions(threshold=sys.maxsize)

from testers import *
from utils import *
import DataGenerator as DG
from torch.utils.data import DataLoader
from control_module import ControlModule
import torchvision.transforms as transforms
np.random.seed(0)
# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
# torch.set_deterministic(True)
# DATA LOADER FOR SINGLE MODALITY
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class data_loader(object):
    def __init__(self, train_val_test,X_lidar_train,X_lidar_test,y_train,y_test):
        if train_val_test == 'train':
            self.feat = X_lidar_train
            self.label = y_train
        elif train_val_test == 'val':
            self.feat = X_lidar_validation
            self.label = y_validation
        elif train_val_test == 'test':
            self.feat = X_lidar_test
            self.label = y_test
        print(train_val_test)

    def __len__(self):
        return self.feat.shape[0]

    def __getitem__(self, index):
        feat = self.feat[index] #
        label = self.label[index] # change
        return torch.from_numpy(feat).type(torch.FloatTensor), torch.from_numpy(label).type(torch.FloatTensor)



class CVTrainValTest():
    def __init__(self, base_path,save_path):

        self.base_path = base_path
        self.save_path = save_path
        print(base_path)
        print(save_path)

    def load_data_cifar(self, batch_size):
        self.x_train = np.asarray(np.load(os.path.join(self.base_path, "train/X.npy")))
        self.y_train = np.asarray(np.load(os.path.join(self.base_path, "train/y.npy")))
        self.x_test = np.asarray(np.load(os.path.join(self.base_path, "test/X.npy")))
        self.y_test = np.asarray(np.load(os.path.join(self.base_path, "test/y.npy")))

        # map label to 0-9
        max_label = np.max(self.y_train)
        if max_label > 9:
            self.y_train = self.y_train - (max_label-9)
            self.y_test = self.y_test - (max_label-9)
        print("# of training exp:%d, testing exp:%d" % (len(self.x_train), len(self.x_test)))

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.training_set = DG.CifarDataGenerator(self.x_train, self.y_train)
        DataParams = {'batch_size': batch_size, 'shuffle': True, 'num_workers':0}
        self.train_generator = DataLoader(self.training_set, **DataParams)

        self.test_set = DG.CifarDataGenerator(self.x_test, self.y_test)
        DataParams = {'batch_size': batch_size, 'shuffle': False, 'num_workers':0}
        self.test_generator = DataLoader(self.test_set, **DataParams)

        return self.train_generator

    def load_data_mnist(self, batch_size):
        self.x_train = np.asarray(np.load(os.path.join(self.base_path, "train/X.npy")))
        self.y_train = np.asarray(np.load(os.path.join(self.base_path, "train/y.npy")))
        self.x_test = np.asarray(np.load(os.path.join(self.base_path, "test/X.npy")))
        self.y_test = np.asarray(np.load(os.path.join(self.base_path, "test/y.npy")))

        # map label to 0-9
        max_label = np.max(self.y_train)
        if max_label > 9:
            self.y_train = self.y_train - (max_label-9)
            self.y_test = self.y_test - (max_label-9)
        print("# of training exp:%d, testing exp:%d" % (len(self.x_train), len(self.x_test)))

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.training_set = DG.MnistDataGenerator(self.x_train, self.y_train)
        DataParams = {'batch_size': batch_size, 'shuffle': True, 'num_workers':0}
        self.train_generator = DataLoader(self.training_set, **DataParams)

        self.test_set = DG.MnistDataGenerator(self.x_test, self.y_test)
        DataParams = {'batch_size': batch_size, 'shuffle': False, 'num_workers':0}
        self.test_generator = DataLoader(self.test_set, **DataParams)

        return self.train_generator

    def load_data_mixture(self, params):
        '''
        Mixture dataset contains 5 tasks, [mnist,cifar,mnist,cifar,mnist]
        Mnist > Cifar => subsample mnist
        # Mnist: 60000
        # Cifar: 5000
        '''
        self.x_train = np.asarray(np.load(os.path.join(self.base_path, "train/X.npy")))
        self.y_train = np.asarray(np.load(os.path.join(self.base_path, "train/y.npy")))
        self.x_test = np.asarray(np.load(os.path.join(self.base_path, "test/X.npy")))
        self.y_test = np.asarray(np.load(os.path.join(self.base_path, "test/y.npy")))

        # map label to 0-9
        max_label = np.max(self.y_train)
        if max_label > 9:
            self.y_train = self.y_train - (max_label-9)
            self.y_test = self.y_test - (max_label-9)
        print("# of training exp:%d, testing exp:%d" % (len(self.x_train), len(self.x_test)))

        # scale number of training sample
        scale = 1
        trigger = False
        if len(self.y_train) > 5000:
            trigger=True
            params.epochs = 50
            params.epochs_prune = 30
            params.epochs_mask_retrain = 50
            print('Sample {} examples in each training epoch.'.format(int(len(self.y_train)*scale)))
        else:
            params.epochs = 300
            params.epochs_prune = 200
            params.epochs_mask_retrain = 300

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.training_set = DG.MixtureDataGenerator(self.x_train, self.y_train, scale=scale, trigger=trigger)
        DataParams = {'batch_size': params.batch_size, 'shuffle': True, 'num_workers':0}
        self.train_generator = DataLoader(self.training_set, **DataParams)

        self.test_set = DG.MixtureDataGenerator(self.x_test, self.y_test, trigger=trigger)
        DataParams = {'batch_size': params.batch_size, 'shuffle': False, 'num_workers':0}
        self.test_generator = DataLoader(self.test_set, **DataParams)
        return params, self.train_generator


    def load_data_flash(self, batch_size):
        self.x_train = np.asarray(np.load(os.path.join(self.base_path, "train/X.npy")))
        # self.y_train =  np.asarray(np.load(os.path.join(self.base_path, "train/y.npy")))
        self.y_train =  np.asarray(np.argmax(np.load(os.path.join(self.base_path, "train/y.npy")), axis=1))
        self.x_test = np.asarray(np.load(os.path.join(self.base_path, "test/X.npy")))
        # self.y_test = np.asarray(np.load(os.path.join(self.base_path, "test/y.npy")))
        self.y_test = np.asarray(np.argmax(np.load(os.path.join(self.base_path, "test/y.npy")), axis=1))

        self.y_train = self.y_train.reshape(self.y_train.shape[0],1)
        self.y_test = self.y_test.reshape(self.y_test.shape[0],1)
        print('self.y_test',self.y_test)
        # print('self.y_test',self.y_test.shape,np.argmax(self.y_test, axis=1),np.argmax(self.y_test, axis=1).shape)

        print('data shapes',self.x_train.shape,self.x_test.shape,self.y_train.shape,self.y_test.shape)
        print('********************padding data********************')

        if True:   ##normalization
            if self.x_train.shape[1]==2:
                # scale = 9747  #bad
                scale = 1  #ok
                print("############GPS scale##########",scale)
            elif self.x_train.shape[1]==45 or self.x_train.shape[1]==90:
                scale = 255
            elif self.x_train.shape[1]==20:
                scale = 1
            else:
                print('invalid data input shape')
            # before = self.x_train[0]
            self.x_train = self.x_train/scale
            self.x_test = self.x_test/scale
            # after = self.x_train[0]
            # print('check equality',(before==after).all())


        if True:   ##padding data
            print('90-self.x_train.shape[1]',self.x_train.shape[1],self.x_train.shape[2],self.x_train.shape[3],90-self.x_train.shape[1],160-self.x_train.shape[2],20-self.x_train.shape[3])
            self.x_train = np.pad(self.x_train, [(0, 0), (0, 45-self.x_train.shape[1]),(0, 80-self.x_train.shape[2]),(0, 20-self.x_train.shape[3])], mode='constant', constant_values=0)
            self.x_test = np.pad(self.x_test, [(0, 0), (0, 45-self.x_test.shape[1]),(0, 80-self.x_test.shape[2]),(0, 20-self.x_test.shape[3])], mode='constant', constant_values=0)
        # elif True and self.x_train.shape[1]==2:  # not a good idea, accuracy stucks at 21
        #     print('90-self.x_train.shape[1]',self.x_train.shape[1],self.x_train.shape[2],self.x_train.shape[3],90-self.x_train.shape[1],160-self.x_train.shape[2],20-self.x_train.shape[3])
        #     self.x_train = np.pad(self.x_train, [(0, 0), (0, 45-self.x_train.shape[1]),(0, 80-self.x_train.shape[2]),(0, 20-self.x_train.shape[3])], mode='constant', constant_values=5)
        #     self.x_test = np.pad(self.x_test, [(0, 0), (0, 45-self.x_test.shape[1]),(0, 80-self.x_test.shape[2]),(0, 20-self.x_test.shape[3])], mode='constant', constant_values=5)

        print('data shapes after',self.x_train.shape,self.x_test.shape,self.y_train.shape,self.y_test.shape)

        self.training_set = data_loader('train',self.x_train,self.x_test,self.y_train,self.y_test)
        self.test_set = data_loader('test',self.x_train,self.x_test,self.y_train,self.y_test)

        self.train_generator = DataLoader(self.training_set, batch_size=32, shuffle=True, num_workers=8,worker_init_fn=seed_worker)
        self.test_generator = DataLoader(self.test_set, batch_size=32, shuffle=False, num_workers=8,worker_init_fn=seed_worker)
        print('self.train_generator',self.train_generator)
        print('self.test_generator',self.test_generator)

        return self.train_generator, self.test_generator



    def train_model(self, args, model, masks, train_loader, criterion, optimizer, scheduler, epoch):
        # print('train func params',args, model, masks)
        atch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        idx_loss_dict = {}
        control = ControlModule(args,model)
        squared_grad_dict = dict()
        #if masks:
        #    test_sparsity_mask(args,masks)
        model.train()
        once = True
        for i, (input, target) in enumerate(train_loader):
            input = input.float().cuda()
            target = target.long().cuda()
            scheduler.step()
            # compute output
            output = model(input)
            # print('output, target',output, target,output.shape, target.shape)  #torch.Size([64, 10]) torch.Size([64])
            ce_loss = criterion(output, target[:,0])

            # measure accuracy and record loss
            prec1,_ = accuracy(output, target, topk=(1,5))
            losses.update(ce_loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            ce_loss.backward()
            # print('masks',masks)
            if masks:    # for task0 args.mask is none
                if once == True:
                    for k in masks.keys():
                        l = masks[k].cpu().detach().numpy()
                        total_size = 1
                        for m in range(len(l.shape)):
                            total_size *=l.shape[m]
                        l =l.reshape(total_size)
                        print("check point",k,l.shape,l[np.nonzero(l)[0]].shape)
                    once = False
                with torch.no_grad():
                    # print("model.named_parameters()",model.named_parameters())
                    for name, W in (model.named_parameters()):
                        # print('name',name)
                        # fixed-layers are shared layers for multi-tasks, it should not be trained besides the first task
                        if name in args.fixed_layer:
                            print("name",name)
                            W.grad *= 0
                            continue
                        if name in masks and name in args.pruned_layer:
                            # print("masks",masks,args.pruned_layer)
                            W.grad *= 1-masks[name].cuda()


            optimizer.step()
            # print("test 2",model.conv1.weight.grad[0])

            # control = ControlModule(model, config=config)
            # for (key, param), g in zip(model.named_parameters(), list_grad):
            #     assert param.size() == g.size()
            #     control.accumulate(key, g ** 2)

            # print(i)
            if i % 100 == 0:
                for param_group in optimizer.param_groups:
                    current_lr = param_group['lr']
                print('({0}) lr:[{1:.5f}]  '
                      'Epoch: [{2}][{3}/{4}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f}% ({top1.avg:.3f}%)\t'
                      .format('adam', current_lr,
                       epoch, i, len(train_loader), loss=losses, top1=top1))
            if i % 100 == 0:
                idx_loss_dict[i] = losses.avg

        return model

    def test_model(self, args, model, test_loader,mask=""):

        """
        Run evaluation
        """
        batch_time = AverageMeter()
        top1 = AverageMeter()

        if mask:
            set_model_mask(model, mask)
        # switch to evaluate mode
        model.eval()

        end = time.time()
        # print('Afterself.test_generator',self.test_generator)
        # for i, (input, target) in enumerate(self.test_generator):
        for i, (input, target) in enumerate(test_loader):
            input = input.float().cuda()
            target = target.long().cuda()

            # compute output
            output = model(input)
            output = output.float()

            # measure accuracy and record loss
            prec1,_ = accuracy(output, target, topk=(1,5))
            top1.update(prec1[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        print('Testing Prec@1 {top1.avg:.3f}%'.format(top1=top1))

        return top1.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
