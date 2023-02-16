import argparse
import glob
import warnings
import os
import random
import numpy as np
import time
import datetime
from pandas.core.common import flatten

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

from models.model import SimpleGLP
from utils.utils import accuracy

#third part
import utils.logging as logging
import utils.metrics as metrics
from utils.criterion import SiLogLoss
from dataset.base_dataset import get_dataset


warnings.filterwarnings(action='ignore')

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=30, help='Number of max epochs in training (default: 100)')
parser.add_argument('--start-epoch', type=int, default=1)
parser.add_argument('--workers', type=int, default=4, help='Number of workers in dataset loader (default: 4)')
parser.add_argument('--batch-size', type=int, default=10, help='Batch size in training (default: 8)')
parser.add_argument('--lr', default=1e-3)

parser.add_argument('--num-classes', type=int, default=3)
parser.add_argument('--crop_h',  type=int, default=448)
parser.add_argument('--crop_w',  type=int, default=576)

parser.add_argument('--log_dir', type=str, default='./logs')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--print-freq', type=int, default=1)
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help="model_args.resume")

# logging options
parser.add_argument('--val_freq', type=int, default=1)
parser.add_argument('--save_freq', type=int, default=1)
#parser.add_argument('--save_model', action='store_true')        
# parser.add_argument('--save_result', action='store_true')

parser.add_argument('--save_model', type=bool, default = True)        #action='store_true'
parser.add_argument('--save_result', type=bool, default = True)
parser.add_argument('--result_dir', type=str, default='./result')

#Depth option
parser.add_argument('--data_path',    type=str, default='./dataset/')
parser.add_argument('--dataset',      type=str, default='nyudepthv2')
parser.add_argument('--exp_name',     type=str, default='test')

# depth configs
parser.add_argument('--max_depth',      type=float, default=10.0)
parser.add_argument('--max_depth_eval', type=float, default=10.0)
parser.add_argument('--min_depth_eval', type=float, default=1e-3)        
parser.add_argument('--do_kb_crop',     type=int, default=1)
parser.add_argument('--kitti_crop', type=str, default=None,
                    choices=['garg_crop', 'eigen_crop'])

# Distributed
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node dataset parallel training')


def main():
    args = parser.parse_args()

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)    

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable dataset parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    exp_name = '%s' % (args.exp_name)
    log_dir = os.path.join(args.log_dir, args.dataset, exp_name)
    logging.check_and_make_dirs(log_dir)
    summary = SummaryWriter(log_dir)
    log_txt = os.path.join(log_dir, 'logs.txt')  
    logging.log_args_to_txt(log_txt, args)

    global result_dir
    result_dir = os.path.join(log_dir, 'results')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    net = SimpleGLP(args.num_classes)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            print("Distributed")
            torch.cuda.set_device(args.gpu)
            net.cuda(args.gpu)
            # discriminator.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])

        else:
            net.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            net = torch.nn.parallel.DistributedDataParallel(net)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        net = net.cuda(args.gpu)

    else:
        net = torch.nn.DataParallel(net).cuda()

    # Optimizer / criterion / scheduler
    criterion = SiLogLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr)
    scheduler = None
    # best_Score = -1e10

    global global_step
    global_step = 0

    # Resume
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map models to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_Score = checkpoint['Score']
            net.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Dataset setting
    dataset_kwargs = {'dataset_name': args.dataset, 'data_path': args.data_path}
    dataset_kwargs['crop_size'] = (args.crop_h, args.crop_w)

    train_dataset = get_dataset(**dataset_kwargs)
    test_dataset = get_dataset(**dataset_kwargs, is_train=False)

    # Sampler
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    # DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),num_workers=args.workers,
                                                pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                               pin_memory=True)

    # Parameters
    param = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total Param: ", param)

    # Train&Validation 
    for epoch in range(args.start_epoch, args.epochs + 1):
        print('\nEpoch: %03d - %03d' % (epoch, args.epochs))
        loss_train = train(train_loader, net, criterion, optimizer=optimizer, 
                            epoch=epoch, scheduler=scheduler ,args=args)
        summary.add_scalar('Training loss', loss_train, epoch)

        if epoch % args.val_freq == 0:
            results_dict, loss_val = validate(test_loader, net, criterion, 
                                              epoch=epoch, args=args)
            summary.add_scalar('Val loss', loss_val, epoch)

            result_lines = logging.display_result(results_dict)

            print(result_lines)

            with open(log_txt, 'a') as txtfile:
                txtfile.write('\nEpoch: %03d - %03d' % (epoch, args.epochs))
                txtfile.write(result_lines)                

            for each_metric, each_results in results_dict.items():
                summary.add_scalar(each_metric, each_results, epoch)



def train(train_loader, net,
          criterion,
          optimizer,
          epoch,
          scheduler,
          args):
    global global_step
    
    net.train()
    end = time.time()
    depth_loss = logging.AverageMeter()

    half_epoch = args.epochs // 2
    
    for batch_idx, batch in enumerate(train_loader):      
        global_step += 1

        for param_group in optimizer.param_groups:
            if global_step < 2019 * half_epoch:
                current_lr = (1e-4 - 3e-5) * (global_step /
                                              2019/half_epoch) ** 0.9 + 3e-5
            else:
                current_lr = (3e-5 - 1e-4) * (global_step /
                                              2019/half_epoch - 1) ** 0.9 + 1e-4
            param_group['lr'] = current_lr

        input_RGB = batch['image'].cuda(args.gpu, non_blocking=True)
        depth_gt = batch['depth'].cuda(args.gpu, non_blocking=True)

        preds = net(input_RGB)

        optimizer.zero_grad()
        loss_d = criterion(preds['pred_d'].squeeze(), depth_gt)
        depth_loss.update(loss_d.item(), input_RGB.size(0))
        loss_d.backward()

        logging.progress_bar(batch_idx, len(train_loader), args.epochs, epoch,
                            ('Depth Loss: %.4f (%.4f)' %
                            (depth_loss.val, depth_loss.avg)))
        optimizer.step()
    
    if scheduler is not None:
        scheduler.step()
    
    elapse = datetime.timedelta(seconds=time.time() - end)
    print(f"걸린 시간: ", elapse)

    return loss_d



def validate(test_loader, net, criterion, epoch, args):

    depth_loss = logging.AverageMeter()
    net.eval()

    metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog']

    if args.save_model:
        torch.save(net.state_dict(), os.path.join(
            args.log_dir, 'epoch_%02d_net.ckpt' % epoch))

    result_metrics = {}
    for metric in metric_name:
        result_metrics[metric] = 0.0

    for batch_idx, batch in enumerate(test_loader):
        input_RGB = batch['image'].cuda(args.gpu, non_blocking=True)
        depth_gt = batch['depth'].cuda(args.gpu, non_blocking=True)
        filename = batch['filename'][0]

        with torch.no_grad():
            preds = net(input_RGB)

        pred_d = preds['pred_d'].squeeze()
        depth_gt = depth_gt.squeeze()

        loss_d = criterion(preds['pred_d'].squeeze(), depth_gt)

        depth_loss.update(loss_d.item(), input_RGB.size(0))

        pred_crop, gt_crop = metrics.cropping_img(args, pred_d, depth_gt)
        computed_result = metrics.eval_depth(pred_crop, gt_crop)
        # save_path = os.path.join(result_dir, filename)
        

        loss_d = depth_loss.avg
        logging.progress_bar(batch_idx, len(test_loader), args.epochs, epoch)

        for key in result_metrics.keys():
            result_metrics[key] += computed_result[key]

    for key in result_metrics.keys():
        result_metrics[key] = result_metrics[key] / (batch_idx + 1)

    return result_metrics, loss_d


if __name__ == "__main__":
    main()
