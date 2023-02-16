import requests
import pickle
import os, random, math
import numpy as np
import cv2

import torch
import torch.nn.functional as F

from enum import Enum
from einops import rearrange, repeat, reduce
from collections import OrderedDict, Counter


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


def quantize(img, n_bits):
    n_colors = 2 ** n_bits
    # Quantize to integers from 0, ..., n_colors - 1
    img = torch.clamp(torch.floor((img * n_colors)), max=n_colors - 1)
    img /= n_colors - 1 # Scale to [0, 1]
    return img


def remove_module_state_dict(state_dict):
    """Clean state_dict keys if original state dict was saved from DistributedDataParallel
       and loaded without"""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    return new_state_dict


def seg_idxs_to_color(segs, palette_fname='palette.pkl'):
    B, H, W = segs.shape

    with open(palette_fname, 'rb') as f:
        palette = pickle.load(f)
    palette = torch.FloatTensor(palette).view(256, 3)
    imgs = torch.index_select(palette, 0, segs.view(-1)).view(B, H, W, 3).permute(0, 3, 1, 2) / 255.
    return imgs


def unnormalize(images, dataset):
    if dataset == 'cifar10':
        mu = [0.4914, 0.4822, 0.4465]
        stddev = [0.2023, 0.1994, 0.2010]
    else:
        mu = [0.485, 0.456, 0.406]
        stddev = [0.229, 0.224, 0.225]

    mu = torch.FloatTensor(mu).view(1, 3, 1, 1)
    stddev = torch.FloatTensor(stddev).view(1, 3, 1, 1)
    return images * stddev + mu


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
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


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
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def save_checkpoint(args, model, optimizer, scheduler, epoch, best_loss, checkpoint_name='best'):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'best_loss': best_loss,
    }

    project = "{}/{}/{}".format(args.arch, args.dataset, args.task)
    root = os.path.join(args.output_dir, project)
    os.makedirs(root, exist_ok=True)
    checkpoint_PATH = os.path.join(root, "%s_checkpoint.ckpt" % checkpoint_name)
    torch.save(state, checkpoint_PATH)
    print("Saved model checkpoint to [DIR: %s]" % checkpoint_PATH)


def load_from_checkpoint(args, model, optimizer, scheduler):
    if args.gpu is None:
        ckpt = torch.load(args.resume_dir)
    else:
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(args.gpu)
        ckpt = torch.load(args.resume_dir, map_location=loc)

    start_epoch = ckpt['epoch']
    best_loss = ckpt['best_loss']
    if args.gpu is not None:
        # best_loss may be from a checkpoint from a different GPU
        best_loss = best_loss.to(args.gpu)

    if args.selfsup:
        model_dict = model.state_dict()
        ckpt['model'] = {k: v for k, v in ckpt['model'].items() if k in model_dict}
        model_dict.update(ckpt['model'])
        model.load_state_dict(model_dict)
        if 'optimizer' in ckpt and 'scheduler' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])

    else:
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        print("Finish to load checkpoint '{}' (epoch {})".format(args.resume_dir, ckpt['epoch']))

    return model, optimizer, start_epoch, best_loss
