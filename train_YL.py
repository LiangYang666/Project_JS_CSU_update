# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from tensorboardX import SummaryWriter
import logging
import os, sys
from tqdm import tqdm
from dataset import Yolo_dataset
from cfg import Cfg
from models_YL import Yolov4
import argparse
from easydict import EasyDict as edict
from torch.nn import functional as F
import ipdb

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    from: https://github.com/chainer/chainercv
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    # top left
    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        # bottom right
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # bottom right
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


def bboxes_giou(bboxes_a, bboxes_b, xyxy=True):
    pass


def bboxes_diou(bboxes_a, bboxes_b, xyxy=True):
    pass


def bboxes_ciou(bboxes_a, bboxes_b, xyxy=True):
    pass


class Yolo_loss(nn.Module):
    def __init__(self, n_classes=80, n_anchors=3, device=None, batch=2):
        super(Yolo_loss, self).__init__()
        self.device = device
        self.strides = [8, 16, 32]
        image_size = 608
        self.n_classes = n_classes
        self.n_anchors = n_anchors

        self.anchors = [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]]
        self.anch_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        self.ignore_thre = 0.5

        self.masked_anchors, self.ref_anchors, self.grid_x, self.grid_y, self.anchor_w, self.anchor_h = [], [], [], [], [], []

        for i in range(3):
            all_anchors_grid = [(w / self.strides[i], h / self.strides[i]) for w, h in self.anchors]
            ''' when i=0
            all_anchors_grid -> [(1.5, 2.0), (2.375, 4.5), (5.0, 3.5), (4.5, 9.375), (9.5, 6.875), (9.0, 18.25), (17.75, 13.75), (24.0, 30.375), (57.375, 50.125)]
            '''
            masked_anchors = np.array([all_anchors_grid[j] for j in self.anch_masks[i]], dtype=np.float32)
            ''' when i=0
            masked_anchors   -> array([[1.5  , 2.   ],
                                       [2.375, 4.5  ],
                                       [5.   , 3.5  ]], dtype=float32)
            '''
            ref_anchors = np.zeros((len(all_anchors_grid), 4), dtype=np.float32)
            ref_anchors[:, 2:] = np.array(all_anchors_grid, dtype=np.float32)
            ref_anchors = torch.from_numpy(ref_anchors)
            '''
            ref_anchors  -> tensor([[ 0.0000,  0.0000,  1.5000,  2.0000],
                                    [ 0.0000,  0.0000,  2.3750,  4.5000],
                                    [ 0.0000,  0.0000,  5.0000,  3.5000],
                                    [ 0.0000,  0.0000,  4.5000,  9.3750],
                                    [ 0.0000,  0.0000,  9.5000,  6.8750],
                                    [ 0.0000,  0.0000,  9.0000, 18.2500],
                                    [ 0.0000,  0.0000, 17.7500, 13.7500],
                                    [ 0.0000,  0.0000, 24.0000, 30.3750],
                                    [ 0.0000,  0.0000, 57.3750, 50.1250]])
            '''
            # calculate pred - xywh obj cls
            fsize = image_size // self.strides[i]   # 76
            grid_x = torch.arange(fsize, dtype=torch.float).repeat(batch, 3, fsize, 1).to(device)
            # when i=0, batch=4, grid_x.size() -> torch.Size([4, 3, 76, 76])
            # (when x=any)  grid_x[x,x,x,:]-> tensor([ 0.,  1.,  2., ..., 73., 74., 75.], device='cuda:0')
            grid_y = torch.arange(fsize, dtype=torch.float).repeat(batch, 3, fsize, 1).permute(0, 1, 3, 2).to(device)
            '''
            (when x=any) 
            grid_y[x,x,...] ->  tensor([[ 0.,  0.,  0.,  ...,  0.,  0.,  0.],
                                        [ 1.,  1.,  1.,  ...,  1.,  1.,  1.],
                                        [ 2.,  2.,  2.,  ...,  2.,  2.,  2.],
                                        ...,
                                        [73., 73., 73.,  ..., 73., 73., 73.],
                                        [74., 74., 74.,  ..., 74., 74., 74.],
                                        [75., 75., 75.,  ..., 75., 75., 75.]], device='cuda:0')
            '''
            anchor_w = torch.from_numpy(masked_anchors[:, 0]).repeat(batch, fsize, fsize, 1).permute(0, 3, 1, 2).to(
                device)
            '''
             anchor_w.size() -> torch.Size([4, 3, 76, 76])  4 is the batch size
             when x = one of (0,1,2,3)
             anchor_w[x]   ->   tensor([[[1.5000, 1.5000, 1.5000,  ..., 1.5000, 1.5000, 1.5000],
                                         [1.5000, 1.5000, 1.5000,  ..., 1.5000, 1.5000, 1.5000],
                                         [1.5000, 1.5000, 1.5000,  ..., 1.5000, 1.5000, 1.5000],
                                         ...,
                                         [1.5000, 1.5000, 1.5000,  ..., 1.5000, 1.5000, 1.5000],
                                         [1.5000, 1.5000, 1.5000,  ..., 1.5000, 1.5000, 1.5000],
                                         [1.5000, 1.5000, 1.5000,  ..., 1.5000, 1.5000, 1.5000]],
                                
                                        [[2.3750, 2.3750, 2.3750,  ..., 2.3750, 2.3750, 2.3750],
                                         [2.3750, 2.3750, 2.3750,  ..., 2.3750, 2.3750, 2.3750],
                                         [2.3750, 2.3750, 2.3750,  ..., 2.3750, 2.3750, 2.3750],
                                         ...,
                                         [2.3750, 2.3750, 2.3750,  ..., 2.3750, 2.3750, 2.3750],
                                         [2.3750, 2.3750, 2.3750,  ..., 2.3750, 2.3750, 2.3750],
                                         [2.3750, 2.3750, 2.3750,  ..., 2.3750, 2.3750, 2.3750]],
                                
                                        [[5.0000, 5.0000, 5.0000,  ..., 5.0000, 5.0000, 5.0000],
                                         [5.0000, 5.0000, 5.0000,  ..., 5.0000, 5.0000, 5.0000],
                                         [5.0000, 5.0000, 5.0000,  ..., 5.0000, 5.0000, 5.0000],
                                         ...,
                                         [5.0000, 5.0000, 5.0000,  ..., 5.0000, 5.0000, 5.0000],
                                         [5.0000, 5.0000, 5.0000,  ..., 5.0000, 5.0000, 5.0000],
                                         [5.0000, 5.0000, 5.0000,  ..., 5.0000, 5.0000, 5.0000]]],
                                       device='cuda:0')

            '''
            anchor_h = torch.from_numpy(masked_anchors[:, 1]).repeat(batch, fsize, fsize, 1).permute(0, 3, 1, 2).to(
                device)
            ''' anchor_h.size() -> torch.Size([4, 3, 76, 76])   # 4 is batch size
            when x = one of (0,1,2,3)
            anchor_h[x] ->  tensor([[[2.0000, 2.0000, 2.0000,  ..., 2.0000, 2.0000, 2.0000],
                                     [2.0000, 2.0000, 2.0000,  ..., 2.0000, 2.0000, 2.0000],
                                     [2.0000, 2.0000, 2.0000,  ..., 2.0000, 2.0000, 2.0000],
                                     ...,
                                     [2.0000, 2.0000, 2.0000,  ..., 2.0000, 2.0000, 2.0000],
                                     [2.0000, 2.0000, 2.0000,  ..., 2.0000, 2.0000, 2.0000],
                                     [2.0000, 2.0000, 2.0000,  ..., 2.0000, 2.0000, 2.0000]],
                            
                                    [[4.5000, 4.5000, 4.5000,  ..., 4.5000, 4.5000, 4.5000],
                                     [4.5000, 4.5000, 4.5000,  ..., 4.5000, 4.5000, 4.5000],
                                     [4.5000, 4.5000, 4.5000,  ..., 4.5000, 4.5000, 4.5000],
                                     ...,
                                     [4.5000, 4.5000, 4.5000,  ..., 4.5000, 4.5000, 4.5000],
                                     [4.5000, 4.5000, 4.5000,  ..., 4.5000, 4.5000, 4.5000],
                                     [4.5000, 4.5000, 4.5000,  ..., 4.5000, 4.5000, 4.5000]],
                            
                                    [[3.5000, 3.5000, 3.5000,  ..., 3.5000, 3.5000, 3.5000],
                                     [3.5000, 3.5000, 3.5000,  ..., 3.5000, 3.5000, 3.5000],
                                     [3.5000, 3.5000, 3.5000,  ..., 3.5000, 3.5000, 3.5000],
                                     ...,
                                     [3.5000, 3.5000, 3.5000,  ..., 3.5000, 3.5000, 3.5000],
                                     [3.5000, 3.5000, 3.5000,  ..., 3.5000, 3.5000, 3.5000],
                                     [3.5000, 3.5000, 3.5000,  ..., 3.5000, 3.5000, 3.5000]]],
                                   device='cuda:0')

            '''

            self.masked_anchors.append(masked_anchors)
            self.ref_anchors.append(ref_anchors)
            self.grid_x.append(grid_x)
            self.grid_y.append(grid_y)
            self.anchor_w.append(anchor_w)
            self.anchor_h.append(anchor_h)

    def build_target(self, pred, labels, batchsize, fsize, n_ch, output_id):
        # target assignment
        tgt_mask = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 4 + self.n_classes).to(device=self.device)
        obj_mask = torch.ones(batchsize, self.n_anchors, fsize, fsize).to(device=self.device)
        tgt_scale = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 2).to(self.device)
        target = torch.zeros(batchsize, self.n_anchors, fsize, fsize, n_ch).to(self.device)

        # labels = labels.cpu().data
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        truth_x_all = (labels[:, :, 2] + labels[:, :, 0]) / (self.strides[output_id] * 2)
        truth_y_all = (labels[:, :, 3] + labels[:, :, 1]) / (self.strides[output_id] * 2)
        truth_w_all = (labels[:, :, 2] - labels[:, :, 0]) / self.strides[output_id]
        truth_h_all = (labels[:, :, 3] - labels[:, :, 1]) / self.strides[output_id]
        truth_i_all = truth_x_all.to(torch.int16).cpu().numpy()
        truth_j_all = truth_y_all.to(torch.int16).cpu().numpy()

        for b in range(batchsize):
            n = int(nlabel[b])
            if n == 0:
                continue
            truth_box = torch.zeros(n, 4).to(self.device)
            truth_box[:n, 2] = truth_w_all[b, :n]
            truth_box[:n, 3] = truth_h_all[b, :n]
            truth_i = truth_i_all[b, :n]    # the grid in i(th) column
            truth_j = truth_j_all[b, :n]    # the grid in j(th) row

            # calculate iou between truth and reference anchors
            anchor_ious_all = bboxes_iou(truth_box.cpu(), self.ref_anchors[output_id])
            best_n_all = anchor_ious_all.argmax(dim=1)
            best_n = best_n_all % 3
            best_n_mask = ((best_n_all == self.anch_masks[output_id][0]) |
                           (best_n_all == self.anch_masks[output_id][1]) |
                           (best_n_all == self.anch_masks[output_id][2]))

            if sum(best_n_mask) == 0:
                continue

            truth_box[:n, 0] = truth_x_all[b, :n]
            truth_box[:n, 1] = truth_y_all[b, :n]

            pred_ious = bboxes_iou(pred[b].view(-1, 4), truth_box, xyxy=False)
            pred_best_iou, _ = pred_ious.max(dim=1)
            pred_best_iou = (pred_best_iou > self.ignore_thre)
            pred_best_iou = pred_best_iou.view(pred[b].shape[:3])
            # set mask to zero (ignore) if pred matches truth
            obj_mask[b] = ~ pred_best_iou

            for ti in range(best_n.shape[0]):
                if best_n_mask[ti] == 1:
                    i, j = truth_i[ti], truth_j[ti]
                    a = best_n[ti]
                    obj_mask[b, a, j, i] = 1
                    tgt_mask[b, a, j, i, :] = 1
                    target[b, a, j, i, 0] = truth_x_all[b, ti] - truth_x_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 1] = truth_y_all[b, ti] - truth_y_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 2] = torch.log(
                        truth_w_all[b, ti] / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 0] + 1e-16)
                    target[b, a, j, i, 3] = torch.log(
                        truth_h_all[b, ti] / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 1] + 1e-16)
                    target[b, a, j, i, 4] = 1
                    target[b, a, j, i, 5 + labels[b, ti, 4].to(torch.int16).cpu().numpy()] = 1
                    tgt_scale[b, a, j, i, :] = torch.sqrt(2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize)
        return obj_mask, tgt_mask, tgt_scale, target

    def forward(self, xin, labels=None):
        loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = 0, 0, 0, 0, 0, 0
        for output_id, output in enumerate(xin):    # xin is a list of length 3, 3 scales of prediction
            batchsize = output.shape[0]
            fsize = output.shape[2]     # feature size
            n_ch = 5 + self.n_classes   # x y w h+confidence+n class

            output = output.view(batchsize, self.n_anchors, n_ch, fsize, fsize)
            output = output.permute(0, 1, 3, 4, 2)  # .contiguous()

            # logistic activation for xy, obj, cls  (no wh?)
            output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(output[..., np.r_[:2, 4:n_ch]])

            pred = output[..., :4].clone()
            pred[..., 0] += self.grid_x[output_id]
            pred[..., 1] += self.grid_y[output_id]
            pred[..., 2] = torch.exp(pred[..., 2]) * self.anchor_w[output_id]
            pred[..., 3] = torch.exp(pred[..., 3]) * self.anchor_h[output_id]

            obj_mask, tgt_mask, tgt_scale, target = self.build_target(pred, labels, batchsize, fsize, n_ch, output_id)

            # loss calculation
            output[..., 4] *= obj_mask
            output[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            output[..., 2:4] *= tgt_scale

            target[..., 4] *= obj_mask
            target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            target[..., 2:4] *= tgt_scale

            loss_xy += F.binary_cross_entropy(input=output[..., :2], target=target[..., :2],
                                              weight=tgt_scale * tgt_scale, size_average=False)
            loss_wh += F.mse_loss(input=output[..., 2:4], target=target[..., 2:4], size_average=False) / 2
            loss_obj += F.binary_cross_entropy(input=output[..., 4], target=target[..., 4], size_average=False)
            loss_cls += F.binary_cross_entropy(input=output[..., 5:], target=target[..., 5:], size_average=False)
            loss_l2 += F.mse_loss(input=output, target=target, size_average=False)

        loss = loss_xy + loss_wh + loss_obj + loss_cls

        return loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2


def collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append([img])
        bboxes.append([box])
    images = np.concatenate(images, axis=0)
    images = images.transpose(0, 3, 1, 2)
    images = torch.from_numpy(images).div(255.0)
    bboxes = np.concatenate(bboxes, axis=0)
    bboxes = torch.from_numpy(bboxes)
    return images, bboxes


def train(model, device, config, epochs=5, batch_size=1, save_cp=True, log_step=2, img_scale=0.5):
    train_dataset = Yolo_dataset(config.train_label, config)
    val_dataset = Yolo_dataset(config.val_label, config)

    n_train = len(train_dataset)
    n_val = len(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=config.batch // config.subdivisions, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=True, collate_fn=collate)

    val_loader = DataLoader(val_dataset, batch_size=config.batch // config.subdivisions, shuffle=True, num_workers=8,
                            pin_memory=True, drop_last=True)

    writer = SummaryWriter(log_dir=config.TRAIN_TENSORBOARD_DIR,
                           filename_suffix=f'OPT_{config.TRAIN_OPTIMIZER}_LR_{config.learning_rate}_BS_{config.batch}_Sub_{config.subdivisions}_Size_{config.width}',
                           comment=f'OPT_{config.TRAIN_OPTIMIZER}_LR_{config.learning_rate}_BS_{config.batch}_Sub_{config.subdivisions}_Size_{config.width}')
    # writer.add_images('legend',
    #                   torch.from_numpy(train_dataset.label2colorlegend2(cfg.DATA_CLASSES).transpose([2, 0, 1])).to(
    #                       device).unsqueeze(0))
    max_itr = config.TRAIN_EPOCHS * n_train
    # global_step = cfg.TRAIN_MINEPOCH * n_train
    global_step = 0
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {config.batch}
        Subdivisions:    {config.subdivisions}
        Learning rate:   {config.learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images size:     {config.width}
        Optimizer:       {config.TRAIN_OPTIMIZER}
        Dataset classes: {config.classes}
        Train label path:{config.train_label}
        Pretrained:
    ''')

    # learning rate setup
    def burnin_schedule(i):
        if i < config.burn_in:
            factor = pow(i / config.burn_in, 4)
        elif i < config.steps[0]:
            factor = 1.0
        elif i < config.steps[1]:
            factor = 0.1
        else:
            factor = 0.01
        return factor

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-08)
    # optimizer = optim.Adam(model.parameters(), lr=config.learning_rate / config.batch, betas=(0.9, 0.999), eps=1e-08)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule)

    ipdb.set_trace()
    criterion = Yolo_loss(device=device, batch=config.batch // config.subdivisions,n_classes=config.classes)
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', verbose=True, patience=6, min_lr=1e-7)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, 0.001, 1e-6, 20)

    model.train()
    lrstr = str(config.learning_rate).split('.')[-1]
    lrstr = '_l'+lrstr  # 学习率有关
    train_info = config.train_label.split('.')[-2][-12:].split('train')[-1]    # 训练样本有关信息
    train_info = lrstr + train_info
    # train_info = 'a'

    for epoch in range(epochs):
        #model.train()
        epoch_loss = 0
        epoch_step = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img', ncols=60) as pbar:
            for i, batch in enumerate(train_loader):
                global_step += 1
                epoch_step += 1
                images = batch[0]
                bboxes = batch[1]

                images = images.to(device=device, dtype=torch.float32)
                bboxes = bboxes.to(device=device)
                # images = images.cuda()
                # bboxes = bboxes.cuda()

                bboxes_pred = model(images)
                ipdb.set_trace()
                loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = criterion(bboxes_pred, bboxes)
                # loss = loss / config.subdivisions
                loss.backward()

                epoch_loss += loss.item()

                if global_step % config.subdivisions == 0:
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()

                if global_step % (log_step * config.subdivisions) == 0:
                    writer.add_scalar('train/Loss', loss.item(), global_step)
                    writer.add_scalar('train/loss_xy', loss_xy.item(), global_step)
                    writer.add_scalar('train/loss_wh', loss_wh.item(), global_step)
                    writer.add_scalar('train/loss_obj', loss_obj.item(), global_step)
                    writer.add_scalar('train/loss_cls', loss_cls.item(), global_step)
                    writer.add_scalar('train/loss_l2', loss_l2.item(), global_step)
                    writer.add_scalar('lr', scheduler.get_lr()[0] * config.batch, global_step)
                    '''
                    pbar.set_postfix({'loss (batch)': loss.item(), 'loss_xy': loss_xy.item(),
                                        'loss_wh': loss_wh.item(),
                                        'loss_obj': loss_obj.item(),
                                        'loss_cls': loss_cls.item(),
                                        'loss_l2': loss_l2.item(),
                                        'lr': scheduler.get_lr()[0] * config.batch
                                        })
                    '''
                    logging.info('Train step_{}: loss : {},loss xy : {},loss wh : {},'
                                  'loss obj : {}，loss cls : {},loss l2 : {},lr : {}'
                                  .format(global_step, loss.item(), loss_xy.item(),
                                          loss_wh.item(), loss_obj.item(),
                                          loss_cls.item(), loss_l2.item(),
                                          scheduler.get_lr()[0] * config.batch))
                    # print('Train step_{}: loss : {},loss xy : {},loss wh : {},'
                    #               'loss obj : {}，loss cls : {},loss l2 : {},lr : {}'
                    #               .format(global_step, loss.item(), loss_xy.item(),
                    #                       loss_wh.item(), loss_obj.item(),
                    #                       loss_cls.item(), loss_l2.item(),
                    #                       scheduler.get_lr()[0] * config.batch))

                pbar.update(images.shape[0])

            if save_cp:
                try:
                    os.mkdir(config.checkpoints)
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
                if epoch == 0:
                    torch.save(model, os.path.join(config.checkpoints, f'Yolov4_epoch_model{epoch + 1}.pth'))
                if (epoch+1) % 25 == 0:
                    torch.save(model.state_dict(), os.path.join(config.checkpoints, f'ckp_chuan_5c{train_info}_ep{epoch + 1}.pth'))
                    logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args(**kwargs):
    cfg = kwargs
    parser = argparse.ArgumentParser(description='Train the Model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=16, help='Batch size', dest='batch')
    parser.add_argument('-s', '--subdivisions', metavar='S', type=int, nargs='?', default=4, help='subdivisions', dest='subdivisions')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='learning_rate')
    parser.add_argument('-f', '--load', dest='load', type=str, default=None,
                        help='Load model from a .pth file')
    parser.add_argument('-g', '--gpu', metavar='G', type=str, default='-1',
                        help='GPU', dest='gpu')
    parser.add_argument('-dir', '--data-dir', type=str, default=None,
                        help='dataset dir', dest='dataset_dir')
    parser.add_argument('-pretrained', type=str, default=None, help='pretrained yolov4.conv.137')
    parser.add_argument('-classes',type=int,default=80,help='dataset classes')
    parser.add_argument('-tlp', '-train_label_path',dest='train_label',type=str,default='train.txt',help="train label path")
    parser.add_argument('-val_label_path', dest='val_label', type=str, default='val.txt', help="valid label path")
    parser.add_argument('-ckp_path', '-checkpoint_label_path', dest='checkpoints', type=str, default='checkpoints', help="checkpoints save path")
    parser.add_argument('-epochs', dest='TRAIN_EPOCHS', type=int, default=10, help="number of training epochs")

    args = vars(parser.parse_args())

    for k in args.keys():
        cfg[k] = args.get(k)

    cfg['checkpoints'] = cfg['dataset_dir'] + '/' + cfg['checkpoints']
    cfg['pretrained'] = cfg['checkpoints'] + '/' + cfg['pretrained']
    cfg['train_label'] = os.path.join(cfg['dataset_dir'],  cfg['train_label'])
    # cfg['train_label'] = os.path.join(cfg['dataset_dir'],  'train10.txt')
    cfg['val_label'] = cfg['dataset_dir']+'/' + cfg['val_label']

    return edict(cfg)


def init_logger(log_file=None, log_dir=None, log_level=logging.INFO, mode='w', stdout=True):
    """
    log_dir: 日志文件的文件夹路径
    mode: 'a', append; 'w', 覆盖原文件写入.
    """
    import datetime
    def get_date_str():
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d_%H-%M-%S')

    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    if log_dir is None:
        log_dir = '~/temp/log/'
    if log_file is None:
        log_file = 'log_' + get_date_str() + '.txt'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, log_file)
    # 此处不能使用logging输出
    print('log file path:' + log_file)

    logging.basicConfig(level=logging.DEBUG,
                        format=fmt,
                        filename=log_file,
                        filemode=mode)

    if stdout:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(log_level)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    return logging


if __name__ == "__main__":
    logging = init_logger(log_dir='log')
    cfg = get_args(**Cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = Yolov4(cfg.pretrained, n_classes=cfg.classes)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device=device)

    try:
        train(model=model,
              config=cfg,
              epochs=cfg.TRAIN_EPOCHS,
              device=device, )
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
