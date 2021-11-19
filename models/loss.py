import torch.nn as nn
import torch
from helpers.torch_helpers import *
from kornia import total_variation
import numpy as np

class FMOLoss(nn.Module):
    def __init__(self, config, ivertices, faces):
        super(FMOLoss, self).__init__()
        self.config = config
        if self.config["loss_normal_weight"] > 0:
            self.smooth_loss = FlattenLoss(faces)
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
                if self.config["do_flip"]:
                    loss2 = fmo_loss(hs_frames_renderer[:,frmi], torch.flip(hs_frames_gt[:,frmi],[1]), self.config)
                    temp_loss,_ = torch.cat((loss1.unsqueeze(0),loss2.unsqueeze(0)),0).min(0)
                else:
                    temp_loss = loss1
                supervised_loss = supervised_loss + temp_loss / hs_frames_renderer.shape[1]
        if not self.config["loss_use_model"]:
            model_loss = supervised_loss

        loss_lap = 0*supervised_loss
        if self.config["predict_vertices"] and self.config["loss_laplacian_weight"] > 0:
            loss_lap = self.config["loss_laplacian_weight"]*self.lapl_loss(vertices)

        loss_ael = 0*supervised_loss
        if self.config["predict_vertices"] and self.config["loss_ael_weight"] > 0:
            loss_ael = self.config["loss_ael_weight"]*get_ael(vertices,faces).mean(1) 

        loss_normal = 0*supervised_loss
        if self.config["predict_vertices"] and self.config["loss_normal_weight"] > 0:
            loss_normal = self.config["loss_normal_weight"]*self.smooth_loss(vertices)

        loss_tv = 0*supervised_loss
        if self.config["loss_total_variation"] > 0:
            texture_maps_ext = torch.cat((texture_maps, texture_maps[:,:,:1]), 2)
            texture_maps_ext2 = torch.cat((texture_maps_ext, texture_maps_ext[:,:,:,:1]), 3)
            loss_tv = self.config["loss_total_variation"]*total_variation(texture_maps_ext2)/(3*(self.config["texture_size"]+1)**2)

        loss = supervised_loss + loss_lap + model_loss + loss_ael + loss_normal + loss_tv
        return supervised_loss, model_loss, loss_ael, loss_lap, loss_normal, loss_tv, loss, model_loss_perframe

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
    # losses = nn.L1Loss(reduction='none')(expected, input_batch[:,:3])
    # model_loss = losses.mean([1,2,3])
    return model_loss, model_loss_perframe


# Taken from
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
        
class FlattenLoss(nn.Module):
    def __init__(self, faces, average=False):
        super(FlattenLoss, self).__init__()
        self.nf = faces.shape[0]
        self.average = average
        
        # faces = faces.detach().cpu().numpy()
        vertices = list(set([tuple(v) for v in np.sort(np.concatenate((faces[:, 0:2], faces[:, 1:3]), axis=0))]))

        v0s = np.array([v[0] for v in vertices], 'int32')
        v1s = np.array([v[1] for v in vertices], 'int32')
        v2s = []
        v3s = []
        for v0, v1 in zip(v0s, v1s):
            count = 0
            for face in faces:
                if v0 in face and v1 in face:
                    v = np.copy(face)
                    v = v[v != v0]
                    v = v[v != v1]
                    if count == 0:
                        v2s.append(int(v[0]))
                        count += 1
                    else:
                        v3s.append(int(v[0]))
        v2s = np.array(v2s, 'int32')
        v3s = np.array(v3s, 'int32')

        self.register_buffer('v0s', torch.from_numpy(v0s).long())
        self.register_buffer('v1s', torch.from_numpy(v1s).long())
        self.register_buffer('v2s', torch.from_numpy(v2s).long())
        self.register_buffer('v3s', torch.from_numpy(v3s).long())

    def forward(self, vertices, eps=1e-6):
        # make v0s, v1s, v2s, v3s
        batch_size = vertices.size(0)

        v0s = vertices[:, self.v0s, :]
        v1s = vertices[:, self.v1s, :]
        v2s = vertices[:, self.v2s, :]
        v3s = vertices[:, self.v3s, :]

        a1 = v1s - v0s
        b1 = v2s - v0s
        a1l2 = a1.pow(2).sum(-1)
        b1l2 = b1.pow(2).sum(-1)
        a1l1 = (a1l2 + eps).sqrt()
        b1l1 = (b1l2 + eps).sqrt()
        ab1 = (a1 * b1).sum(-1)
        cos1 = ab1 / (a1l1 * b1l1 + eps)
        sin1 = (1 - cos1.pow(2) + eps).sqrt()
        c1 = a1 * (ab1 / (a1l2 + eps))[:, :, None]
        cb1 = b1 - c1
        cb1l1 = b1l1 * sin1

        a2 = v1s - v0s
        b2 = v3s - v0s
        a2l2 = a2.pow(2).sum(-1)
        b2l2 = b2.pow(2).sum(-1)
        a2l1 = (a2l2 + eps).sqrt()
        b2l1 = (b2l2 + eps).sqrt()
        ab2 = (a2 * b2).sum(-1)
        cos2 = ab2 / (a2l1 * b2l1 + eps)
        sin2 = (1 - cos2.pow(2) + eps).sqrt()
        c2 = a2 * (ab2 / (a2l2 + eps))[:, :, None]
        cb2 = b2 - c2
        cb2l1 = b2l1 * sin2

        cos = (cb1 * cb2).sum(-1) / (cb1l1 * cb2l1 + eps)

        dims = tuple(range(cos.ndimension())[1:])
        loss = (cos + 1).pow(2).mean(dims)
        if self.average:
            return loss.sum() / batch_size
        else:
            return loss