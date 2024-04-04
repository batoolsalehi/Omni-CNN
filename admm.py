from __future__ import print_function
import torch
torch.manual_seed(1)
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
import operator
from numpy import linalg as LA
import yaml
import random
random.seed(0)
from testers import *
from itertools import combinations
import collections
import time
from tqdm import tqdm
# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
# torch.set_deterministic(True)
class ADMM:
    def __init__(self, args, model, config, rho=0.001):
        self.ADMM_U = {}
        self.ADMM_Z = {}
        self.rho = rho
        self.rhos = {}

        if args.adaptive_mask:
            self.ADMM_K = {}
            self.ADMM_Y = {}
            self.rhos_mask = {}

        self.init(args, config, model)

    def init(self, args, config, model):
        """
        Args:
            config: configuration file that has settings for prune ratios, rhos
        called by ADMM constructor. config should be a .yaml file

        """

        if not config:
            raise Exception("must provide a config setting.")

        if not isinstance(config, str):
            self.prune_ratios = config
            for k, v in config.items():
                self.rhos[k] = self.rho
        else:
            with open(config, "r") as stream:
                try:
                    raw_dict = yaml.load(stream)
                    self.prune_ratios = raw_dict['prune_ratios']
                    for k, v in self.prune_ratios.items():
                        self.rhos[k] = self.rho
                except yaml.YAMLError as exc:
                    print(exc)


        for (name, W) in model.named_parameters():
            if name in self.prune_ratios:
                self.ADMM_U[name] = torch.zeros(W.shape).cuda()  # add U
                self.ADMM_Z[name] = torch.Tensor(W.shape).cuda()  # add Z
                if args.admm == True and args.dynamic==True: #and name=="conv1.weight": # dont run again during fine-tuning+if dynamic pruning
                    # print("checking grads",model.conv1.weight.grad)
                    self.prune_ratios[name] = find_layers_pruning_ratio(args,name, W)

            elif 'mask' in name and args.adaptive_mask:
                self.rhos_mask[name] = self.rho
                self.ADMM_K[name] = torch.zeros(W.shape).cuda()  # add K
                self.ADMM_Y[name] = torch.Tensor(W.shape).cuda()  # add Y
            # if(len(W.size()) == 4):
            #     if name not in self.prune_ratios:
            #         continue
            #     self.ADMM_U[name] = torch.zeros(W.shape).cuda()  # add U
            #     self.ADMM_Z[name] = torch.Tensor(W.shape).cuda()  # add Z

#### metric 1
def find_layers_pruning_ratio(args,key,param):  # in np array form
    print("*************find_layers_pruning_ratio**************")
    squared_grad_dict = dict()
    prune_ratios = {}
    #######################
    # for key, param in (model.named_parameters()):
    # print("********key",key)
    g = param.grad
    sgrad = g ** 2
    # print("sgrad,param",param,len(np.nonzero(param.reshape(32*45*3*3).cpu().detach().numpy())[0]),sgrad)
    assert param.size() == g.size()
    if key in squared_grad_dict.keys():
        squared_grad_dict[key] += sgrad
    else:
        squared_grad_dict[key] = sgrad
    #######################get totoal number of elements
    print("get totoal number of elements")
    # np.save("logs_new/grads_"+key+"_.npy",squared_grad_dict[key].cpu())
    total_size = 1
    for l in range(len(squared_grad_dict[key].shape)):
        total_size *=squared_grad_dict[key].shape[l]
    all_elements = squared_grad_dict[key].reshape(total_size)
    print("all_elements",all_elements.shape,all_elements[:10],type(all_elements))
    print("check if torch nonzero is the same as np",torch.nonzero(all_elements).shape)
    all_elements_np = all_elements.cpu().detach().numpy()
    print("np",type(all_elements_np),all_elements_np.shape)
    non_zero_elements = all_elements_np[np.nonzero(all_elements_np)[0]]
    print("non_zero_elements",non_zero_elements.shape,non_zero_elements[:10])   #non_zero_elements torch.Size([5654, 1])  #ok
    ###########################sort and calcuate division
    print("sort and calcuate division")
    values = np.zeros((len(non_zero_elements),), dtype=float)
    sorted_non_zero_elements = -np.sort(-non_zero_elements)
    print("len(sorted_non_zero_elements)",type(sorted_non_zero_elements),sorted_non_zero_elements.shape,sorted_non_zero_elements[:10])

    # if key!="hidden1.weight":
    #     for e in tqdm(range(len(sorted_non_zero_elements))):
    #         values[e] = sum(sorted_non_zero_elements[:e+1])/(e+1)
    # else:  # for speed
    #     for e in tqdm(range(8598*2)):
    #         values[e] = sum(sorted_non_zero_elements[:e+1])/(e+1)
    for e in tqdm(range(len(sorted_non_zero_elements))):
        values[e] = sum(sorted_non_zero_elements[:e+1])/(e+1)
    ###########################check prune ratio
    # if all(v == 0 for v in values):
    #     print("##########using fixed pruning ratios############")
    #     for_GPS = {"conv1.weight":0.2222222222222222,"conv2.weight":0.7887527036770008,"conv3.weight":0.6973046344774094,"conv4.weight":0.7774899251583189,"conv5.weight":0.40034071550255534,"conv6.weight":0.7579214195183777,"conv7.weight":0.4367176634214186,"conv8.weight":0.7061085972850678,"conv9.weight":0.363767518549052,"hidden1.weight":0.9546323583779616,"hidden3.weight":0.5071961068544212,"out.weight":0.7114281400966184}
    #     return for_GPS[key]
    # print("check prune ratio",values.shape,values[0:10],values[-1],len(values))
    for i in range(1,len(values)):
        # print("values[i],values[-1]",values[i],values[-1]+(values[0]-values[-1])*0.02,values[-1])
        if values[i] < values[-1]+(values[0]-values[-1])*args.sensetivity:               #values[-1]+values[-1]*0.1:
            # print("selected pruning ratio",i/len(all_elements),key)
            # self.prune_ratios[key] = i/len(sorted_non_zero_elements)
            # selected_pruning_ratio = 1-(i/len(sorted_non_zero_elements)) #1
            selected_pruning_ratio = 1-(i/len(all_elements_np))  #2
            print("key,pruning ratio",key,i,selected_pruning_ratio)
            break
    return selected_pruning_ratio


def random_pruning(args, weight, prune_ratio):
    weight = weight.cpu().detach().numpy()  # convert cpu tensor to numpy

    if (args.sparsity_type == "filter"):
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        indices = np.random.choice(shape2d[0], int(shape2d[0] * prune_ratio), replace=False)
        weight2d[indices, :] = 0
        weight = weight2d.reshape(shape)
        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
        for i in range(shape2d[0]):
            expand_above_threshold[i, :] = i not in indices
        weight = weight2d.reshape(shape)
        expand_above_threshold = expand_above_threshold.reshape(shape)
        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()
    else:
        raise Exception("not implemented yet")


def L1_pruning(args, weight, prune_ratio):
    """
    projected gradient descent for comparison

    """
    percent = prune_ratio * 100
    weight = weight.cpu().detach().numpy()  # convert cpu tensor to numpy
    shape = weight.shape
    weight2d = weight.reshape(shape[0], -1)
    shape2d = weight2d.shape
    row_l1_norm = LA.norm(weight2d, 1, axis=1)
    percentile = np.percentile(row_l1_norm, percent)
    under_threshold = row_l1_norm < percentile
    above_threshold = row_l1_norm > percentile
    weight2d[under_threshold, :] = 0
    above_threshold = above_threshold.astype(np.float32)
    expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
    for i in range(shape2d[0]):
        expand_above_threshold[i, :] = above_threshold[i]
    weight = weight.reshape(shape)
    expand_above_threshold = expand_above_threshold.reshape(shape)
    return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()


def weight_pruning(args, weight, prune_ratio):
    """
    weight pruning [irregular,column,filter]
    Args:
         weight (pytorch tensor): weight tensor, ordered by output_channel, intput_channel, kernel width and kernel height
         prune_ratio (float between 0-1): target sparsity of weights

    Returns:
         mask for nonzero weights used for retraining
         a pytorch tensor whose elements/column/row that have lowest l2 norms(equivalent to absolute weight here) are set to zero

    """

    weight = weight.cpu().detach().numpy()  # convert cpu tensor to numpy

    percent = prune_ratio * 100
    if (args.sparsity_type == "irregular"):
        weight_temp = np.abs(weight)  # a buffer that holds weights with absolute values
        percentile = np.percentile(weight_temp, percent)  # get a value for this percentitle
        under_threshold = weight_temp < percentile
        above_threshold = weight_temp > percentile
        above_threshold = above_threshold.astype(
            np.float32)  # has to convert bool to float32 for numpy-tensor conversion
        weight[under_threshold] = 0
        return torch.from_numpy(above_threshold).cuda(), torch.from_numpy(weight).cuda()
    else:
        raise SyntaxError("Unknown sparsity type")

def hard_prune(args, ADMM, model, option=None):
    """
    hard_pruning, or direct masking
    Args:
         model: contains weight tensors in cuda
         submask: updated mask.
         mask: the cummulative mask up until now.

    """

    print("hard pruning")

    sparsity_type = args.sparsity_type

    for (name, W) in model.named_parameters():
        print('ADMM.prune_ratios',ADMM.prune_ratios)
        if name in ADMM.prune_ratios:  # ignore layers that do not have rho
            # Set irregular type only for fc layers
            if 'linear' in name or 'fc' in name:
                args.sparsity_type = 'irregular'

            cuda_pruned_weights = None
            if option == None:
                _, cuda_pruned_weights = weight_pruning(args, W, ADMM.prune_ratios[name])  # get sparse model in cuda
            elif option == "random":
                _, cuda_pruned_weights = random_pruning(args, W, ADMM.prune_ratios[name])

            elif option == "l1":
                _, cuda_pruned_weights = L1_pruning(args, W, ADMM.prune_ratios[name])
            else:
                raise Exception("not implmented yet")
            W.data = cuda_pruned_weights  # replace the data field in variable

            args.sparsity_type = sparsity_type

def hard_prune_mask(args, ADMM, model, option=None):
    """
    hard_pruning, or direct masking
    Args:
         model: contains weight tensors in cuda
         submask: updated mask.
         mask: the cummulative mask up until now.

    """

    print("Mask hard pruning")

    sparsity_type = args.sparsity_type

    for (name, W) in model.named_parameters():
        if 'mask' in name:
            W.data = mask_pruning(args.adaptive_ratio, W, args.mask[name.replace('w_mask', 'weight')])
    '''
    mask = {}
    for (name, W) in model.named_parameters():
        if 'mask' in name:
            mask[name] = W
    threshold = find_threshold(args.adaptive_ratio, mask, args.mask)
    for (name, W) in model.named_parameters():
        if 'mask' in name:
            W.data = mask_pruning(threshold, W, args.mask[name.replace('w_mask', 'weight')])
    '''
def mask_pruning(ratio, submask, mask):
    '''
    project mask to M
    Args:
        submask: updated mask.
        mask: the cummulative mask up until now.
    return:
        a subset of the cummulative mask up until now.
    '''
    submask = submask.cpu().detach().numpy()  # convert cpu tensor to numpy
    mask = mask.cpu().detach().numpy()
    mask_temp = submask * mask  # a buffer that holds weights with absolute values


    percent = (1-ratio) * 100
    percentile = np.percentile(mask_temp[mask_temp!=0], percent)  # get a value for this percentitle

    #percentile = ratio
    #print(percentile)

    above_threshold = mask_temp >= percentile
    under_threshold = mask_temp < percentile

    submask[above_threshold] = 1
    submask[under_threshold] = 0
    return torch.from_numpy(submask).cuda()

def find_threshold(ratio, submask, mask):
    All = np.array([])
    for (name, W) in submask.items():
        W = W.cpu().detach().numpy()
        M = mask[name.replace('w_mask', 'weight')].cpu().detach().numpy()
        W = W * M
        W = W.view()
        All = np.concatenate((All, W), axis=None) if All.size else W
    percent = (1-ratio) * 100
    percentile = np.percentile(All[All!=0], percent)
    return percentile

def admm_initialization(args, ADMM, model):
    for i, (name, W) in enumerate(model.named_parameters()):
        if args.admm and name in ADMM.prune_ratios:
            print("^^^^^^^^^^^^^^^^^^^",name,ADMM.prune_ratios[name])
            _, updated_Z = weight_pruning(args, W, ADMM.prune_ratios[name])  # Z(k+1) = W(k+1)+U(k)  U(k) is zeros her
            ADMM.ADMM_Z[name] = updated_Z

        if args.adaptive_mask and args.mask and 'mask' in name:
            weight = W.cpu().detach()
            ADMM.ADMM_Y[name] = mask_pruning(args.adaptive_ratio, weight, args.mask[name.replace('w_mask', 'weight')])





def z_u_update(args, ADMM, model, device, train_loader, optimizer, epoch, data, batch_idx, writer):
    # print("^^^^^^^^^^^^^^z_u_update^^^^^^^^^^^^^^^^^^^^^")
    squared_grad_dict = dict()
    if not args.admm:
        return
    if epoch != 1 and (epoch - 1) % args.admm_epochs == 0 and batch_idx == 0:
        for i, (name, W) in enumerate(model.named_parameters()):
            if name not in ADMM.prune_ratios:
                continue
            Z_prev = None
            if (args.verbose):
                Z_prev = torch.Tensor(ADMM.ADMM_Z[name].cpu()).cuda()
            ####new_rho
            # print('config in z_u_update',config)
            # if args['multi_rho']: #############added
            print('args.multi_rho',args.multi_rho)
            if args.multi_rho: #############added
                print('running admm_multi_rho_scheduler')
                admm_multi_rho_scheduler(ADMM,name) # call multi rho scheduler every admm update


            ADMM.ADMM_Z[name] = W + ADMM.ADMM_U[name]  # Z(k+1) = W(k+1)+U[k]

            _, updated_Z = weight_pruning(args, ADMM.ADMM_Z[name], ADMM.prune_ratios[name])  # equivalent to Euclidean Projection
            ADMM.ADMM_Z[name] = updated_Z
            if (args.verbose):
                if writer:
                    writer.add_scalar('layer:{} W(k+1)-Z(k+1)'.format(name), torch.sqrt(torch.sum((W - ADMM.ADMM_Z[name]) ** 2)).item(), epoch)
                    writer.add_scalar('layer:{} Z(k+1)-Z(k)'.format(name), torch.sqrt(torch.sum((ADMM.ADMM_Z[name] - Z_prev) ** 2)).item(), epoch)

            ADMM.ADMM_U[name] = W - ADMM.ADMM_Z[name] + ADMM.ADMM_U[name]  # U(k+1) = W(k+1) - Z(k+1) +U(k)

def y_k_update(args, ADMM, model, device, train_loader, optimizer, epoch, data, batch_idx, writer):
    if not args.admm_mask:
        return
    if epoch != 1 and (epoch - 1) % args.mask_admm_epochs == 0 and batch_idx == 0:
        for i, (name, M) in enumerate(model.named_parameters()):
            if 'mask' not in name:
                continue
            ADMM.ADMM_Y[name] = M + ADMM.ADMM_K[name]  # Z(k+1) = W(k+1)+U[k]
            ADMM.ADMM_Y[name] = mask_pruning(args.adaptive_ratio, ADMM.ADMM_Y[name], args.mask[name.replace('w_mask', 'weight')])
            ADMM.ADMM_K[name] = M - ADMM.ADMM_Y[name] + ADMM.ADMM_K[name]  # U(k+1) = W(k+1) - Z(k+1) +U(k)

        #threshold = find_threshold(args.adaptive_ratio, ADMM.ADMM_Y, args.mask)
        #for i, (name, M) in enumerate(model.named_parameters()):
        #    if 'mask' not in name:
        #        continue
        #    ADMM.ADMM_Y[name] = mask_pruning(threshold, ADMM.ADMM_Y[name], args.mask[name.replace('w_mask', 'weight')])
        #    ADMM.ADMM_K[name] = M - ADMM.ADMM_Y[name] + ADMM.ADMM_K[name]  # U(k+1) = W(k+1) - Z(k+1) +U(k)

def append_admm_loss(args, ADMM, model, ce_loss):
    '''
    append admm loss to cross_entropy loss
    Args:
        args: configuration parameters
        model: instance to the model class
        ce_loss: the cross entropy loss
    Returns:
        ce_loss(tensor scalar): original cross enropy loss
        admm_loss(dict, name->tensor scalar): a dictionary to show loss for each layer
        ret_loss(scalar): the mixed overall loss

    '''
    admm_loss = {}


    for i, (name, W) in enumerate(model.named_parameters()):  ## initialize Z (for both weights and bias)
        if name in ADMM.prune_ratios:
            admm_loss[name] = 0.5 * ADMM.rhos[name] * (torch.norm(W - ADMM.ADMM_Z[name] + ADMM.ADMM_U[name], p=2) ** 2)

    mixed_loss = 0
    mixed_loss += ce_loss
    for k, v in admm_loss.items():
        mixed_loss += v
    return ce_loss, admm_loss, mixed_loss

def append_mask_loss(args, ADMM, model, ce_loss):
    '''
    append admm loss to cross_entropy loss
    Args:
        args: configuration parameters
        model: instance to the model class
        ce_loss: the cross entropy loss
    Returns:
        ce_loss(tensor scalar): original cross enropy loss
        admm_loss(dict, name->tensor scalar): a dictionary to show loss for each layer
        ret_loss(scalar): the mixed overall loss

    '''
    admm_loss = {}

    for i, (name, W) in enumerate(model.named_parameters()):  ## initialize Z (for both weights and bias)
        if 'mask' in name:
            value = 0.5 * ADMM.rhos_mask[name] * (torch.norm(W - ADMM.ADMM_Y[name] + ADMM.ADMM_K[name], p=2) ** 2)
            admm_loss[name] = 0.5 * ADMM.rhos_mask[name] * (torch.norm(W - ADMM.ADMM_Y[name] + ADMM.ADMM_K[name], p=2) ** 2)


    mixed_loss = 0
    mixed_loss += ce_loss
    for k, v in admm_loss.items():
        mixed_loss += v
    return ce_loss, admm_loss, mixed_loss

class CrossEntropyLossMaybeSmooth(nn.CrossEntropyLoss):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    def __init__(self, smooth_eps=0.0):
        super(CrossEntropyLossMaybeSmooth, self).__init__()
        self.smooth_eps = smooth_eps

    def forward(self, output, target, smooth=False):
        if not smooth:
            return F.cross_entropy(output, target)

        target = target.contiguous().view(-1)
        n_class = output.size(1)
        one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
        smooth_one_hot = one_hot * (1 - self.smooth_eps) + (1 - one_hot) * self.smooth_eps / (n_class - 1)
        log_prb = F.log_softmax(output, dim=1)
        loss = -(smooth_one_hot * log_prb).sum(dim=1).mean()
        return loss


def mixup_data(x, y, alpha=1.0):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam, smooth):
    return lam * criterion(pred, y_a, smooth=smooth) + \
           (1 - lam) * criterion(pred, y_b, smooth=smooth)

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_iter: target learning rate is reached at total_iter, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_iter, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_iter = total_iter
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_iter:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_iter + 1.) for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished and self.after_scheduler:
            return self.after_scheduler.step(epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)





def admm_multi_rho_scheduler(ADMM, name):
    """
    It works better to make rho monotonically increasing
    rho: using 1.1:
           0.01   ->  50epochs -> 1
           0.0001 -> 100epochs -> 1
         using 1.2:
           0.01   -> 25epochs -> 1
           0.0001 -> 50epochs -> 1
         using 1.3:
           0.001   -> 25epochs -> 1
         using 1.6:
           0.001   -> 16epochs -> 1
    """
    current_rho = ADMM.rhos[name]
    print('current rho for new_rho',current_rho)
    ADMM.rhos[name] = min(1, 1.1*current_rho)  # choose whatever you like

def admm_adjust_learning_rate(optimizer, epoch, args):
    """ (The pytorch learning rate scheduler)
Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    """
    For admm, the learning rate change is periodic.
    When epoch is dividable by admm_epoch, the learning rate is reset
    to the original one, and decay every 3 epoch (as the default
    admm epoch is 9)

    """
    # admm_epoch = args['admm_epochs'] ###upadted
    admm_epoch = args.admm_epochs

    lr = None
    if (epoch - 1) % admm_epoch == 0:
        lr = args.lr  ##args['lr']
    else:
        admm_epoch_offset = (epoch - 1) % admm_epoch

        admm_step = admm_epoch / 3  # roughly every 1/3 admm_epoch.

        # lr = args['lr'] * (0.1**(admm_epoch_offset // admm_step))
        lr = args.lr * (0.1**(admm_epoch_offset // admm_step))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


