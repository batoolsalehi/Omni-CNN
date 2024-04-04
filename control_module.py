import torch
from timeit import default_timer as timer
import numpy as np

def main_control(args,model, squared_grad_dict: dict):
    sum_sqg = 0
    # sum_time = config.TIME_CONSTANT
    list_tba_values, list_tba_indices = [], []
    list_coefficient = []
    proc_start = timer()


    if masks:    # for task0 args.mask is none
        for name, W in (model.named_parameters()):
            # fixed-layers are shared layers for multi-tasks, it should not be trained besides the first task
            if name in masks and name in args.pruned_layer:
                W.grad *= 1-masks[name].cuda()



class ControlModule:
    def __init__(self, args, model):
        self.model = model
        self.squared_grad_dict = dict()

    @torch.no_grad()
    def accumulate(self, key, sgrad):
        if key in self.squared_grad_dict.keys():
            self.squared_grad_dict[key] += sgrad
        else:
            self.squared_grad_dict[key] = sgrad

        # np.save("layer_"+str(key)+"_grad",self.squared_grad_dict[key].cpu())

    def adjust(self, args,dec_thr_pct, max_density=None):
        print("squared_grad_dict",self.squared_grad_dict.keys(),self.squared_grad_dict['conv1.weight'].shape)
        main_control(args,self.model, self.squared_grad_dict, dec_thr_pct, max_density)
        self.squared_grad_dict = dict()
