import torch.nn as nn
import torch
from kornia import total_variation
import numpy as np

class FMOLoss(nn.Module):
    def __init__(self, config, ivertices, faces):
        super(FMOLoss, self).__init__()
        self.config = config
        if self.config["loss_laplacian_weight"] > 0:    
            self.lapl_loss = LaplacianLoss(ivertices, faces)

    def forward(self, renders, hs_frames, input_batch, translation, quaternion, vertices, texture_maps, faces):
        model_loss = 0
        model_loss_perframe = []
        if self.config["loss_use_model"]:
            modelled_renders = torch.cat( (renders[:,:,:,:3]*renders[:,:,:,3:], renders[:,:,:,3:4]), 3).mean(2)

            region_of_interest = None
            if self.config["loss_use_supervised"]:
                region_of_interest = hs_frames[:,:,:,3:].mean(2) > 0.05
            
            model_loss, model_loss_perframe  = fmo_model_loss(input_batch, modelled_renders, Mask = region_of_interest)
        
        supervised_loss = 0*model_loss
        if self.config["loss_use_supervised"]:
            if hs_frames[:,:,:,3].min() == 1.0:
                hs_frames_renderer = torch.cat((input_batch[:,:,None,3:] * (1 - renders[:,:,:,3:]) + renders[:,:,:,:3], renders[:,:,:,3:]), 3)
                hs_frames_gt = torch.cat((hs_frames[:,:,:,:3], hs_frames[:,:,:,3:]), 3)
            else:
                hs_frames_renderer = renders
                hs_frames_gt = hs_frames
            for frmi in range(hs_frames_renderer.shape[1]):
                loss1 = fmo_loss(hs_frames_renderer[:,frmi], hs_frames_gt[:,frmi], self.config)
                supervised_loss = supervised_loss + loss1 / hs_frames_renderer.shape[1]
        if not self.config["loss_use_model"]:
            model_loss = supervised_loss

        loss_lap = 0*supervised_loss
        if self.config["predict_vertices"] and self.config["loss_laplacian_weight"] > 0:
            loss_lap = self.config["loss_laplacian_weight"]*self.lapl_loss(vertices)

        loss_tv = 0*supervised_loss
        if self.config["loss_total_variation"] > 0:
            texture_maps_ext = torch.cat((texture_maps, texture_maps[:,:,:1]), 2)
            texture_maps_ext2 = torch.cat((texture_maps_ext, texture_maps_ext[:,:,:,:1]), 3)
            loss_tv = self.config["loss_total_variation"]*total_variation(texture_maps_ext2)/(3*(self.config["texture_size"]+1)**2)

        loss = supervised_loss + loss_lap + model_loss + loss_tv
        return supervised_loss, model_loss, loss_lap, loss_tv, loss, model_loss_perframe

def fmo_loss(Yp, Y, config):
    YM = Y[:,:,-1:,:,:]
    YpM = Yp[:,:,-1:,:,:]
    YF = Y[:,:,:3]
    YpF = Yp[:,:,:3]
    YMb = ((YM+YpM) > 0).type(YpM.dtype)

    loss = torch.zeros((YM.shape[0],1)).to(Y.device)
    if config["loss_mask_weight"] > 0:
        loss = loss + config["loss_mask_weight"]*batch_loss(YpM, YM, YMb, do_mult=False)

    if config["loss_jointm_iou_weight"] > 0:
        loss = loss + config["loss_jointm_iou_weight"]*iou_loss(YM.max(1)[0][:,None], YpM.max(1)[0][:,None])

    if config["loss_rgb_weight"] > 0:
        loss = loss + config["loss_rgb_weight"]*batch_loss(YpF, YF*YM, YMb[:,:,[0,0,0]], do_mult=False)
    
    if config["loss_iou_weight"] > 0:
        loss = loss + config["loss_iou_weight"]*iou_loss(YM, YpM)

    return loss

def iou_loss(YM, YpM):
    A_inter_B = YM * YpM
    A_union_B = (YM + YpM - A_inter_B)
    iou = 1 - (torch.sum(A_inter_B, [2,3,4]) / torch.sum(A_union_B, [2,3,4])).mean(1)
    return iou

def batch_loss(YpM, YM, YMb, do_mult=True):
    if do_mult:
        losses = nn.L1Loss(reduction='none')(YpM*YMb, YM*YMb)
    else:
        losses = nn.L1Loss(reduction='none')(YpM, YM)
    if len(losses.shape) > 4:
        bloss = losses.sum([1,2,3,4]) / YMb.sum([1,2,3,4])
    else:
        bloss = losses.sum([1,2,3]) / (YMb.sum([1,2,3]) + 0.01)
    return bloss

    
def fmo_model_loss(input_batch, renders, Mask = None):    
    expected = input_batch[:,:,3:] * (1 - renders[:,:,3:]) + renders[:,:,:3]
    if Mask is None:
        Mask = renders[:,:,3:] > 0.05
    Mask = Mask.type(renders.dtype)
    model_loss = 0
    model_loss_perframe = []
    for frmi in range(input_batch.shape[1]):
        temp_loss = batch_loss(expected[:,frmi], input_batch[:,frmi,:3], Mask[:,frmi])
        model_loss_perframe.append(temp_loss.item())
        model_loss = model_loss + temp_loss / input_batch.shape[1]
    return model_loss, model_loss_perframe


# https://github.com/ShichenLiu/SoftRas/blob/master/soft_renderer/losses.py
class LaplacianLoss(nn.Module):
    def __init__(self, vertex, faces, average=False):
        super(LaplacianLoss, self).__init__()
        self.nv = vertex.shape[0]
        self.nf = faces.shape[0]
        self.average = average
        laplacian = np.zeros([self.nv, self.nv]).astype(np.float32)

        laplacian[faces[:, 0], faces[:, 1]] = -1
        laplacian[faces[:, 1], faces[:, 0]] = -1
        laplacian[faces[:, 1], faces[:, 2]] = -1
        laplacian[faces[:, 2], faces[:, 1]] = -1
        laplacian[faces[:, 2], faces[:, 0]] = -1
        laplacian[faces[:, 0], faces[:, 2]] = -1

        r, c = np.diag_indices(laplacian.shape[0])
        laplacian[r, c] = -laplacian.sum(1)

        for i in range(self.nv):
            laplacian[i, :] /= laplacian[i, i]

        self.register_buffer('laplacian', torch.from_numpy(laplacian))

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.matmul(self.laplacian, x)
        dims = tuple(range(x.ndimension())[1:])
        x = x.pow(2).mean(dims)
        if self.average:
            return x.sum() / batch_size
        else:
            return x
