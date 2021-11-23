import os
import torch
import numpy as np

from utils import *
from models.initial_mesh import generate_initial_mesh
from models.kaolin_wrapper import load_obj

from models.encoder import *
from models.rendering import *
from models.loss import *

from kornia.feature import DeFMO
from torchvision import transforms


class MotionFromBlur():
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.defmo = DeFMO(pretrained=True).to(device)
        self.defmo.train(False)

    def apply(self,Is,Bs,bboxs_tight,nsplits,radius,obj_dim):
        g_resolution_x = int(640/2)
        g_resolution_y = int(480/2)
        self.defmo.rendering.tsr_steps = nsplits*self.config["factor"]
        self.defmo.rendering.times = torch.linspace(0.01,0.99,nsplits*self.config["factor"])
        self.config["input_frames"] = len(Is)
        self.bbox = bboxs_tight[0].copy()
        for bbox_tight in bboxs_tight:
            bbox = extend_bbox(bbox_tight.copy(),1.0*np.max(radius),g_resolution_y/g_resolution_x,Is[0].shape)
            # bbox = bbox_tight.copy()
            self.bbox[:2] = np.c_[self.bbox[:2], bbox[:2]].min(1)
            self.bbox[2:] = np.c_[self.bbox[2:], bbox[2:]].max(1)
        # self.bbox = extend_bbox_uniform(self.bbox.copy(),0.5*np.max(radius),Is[0].shape)
        input_batch_sfb = torch.Tensor([])
        hs_frames_sfb = torch.Tensor([])
        new_res = ((self.bbox[3]-self.bbox[1]), (self.bbox[2]-self.bbox[0]))
        for I, B, bbox_tight in zip(Is,Bs,bboxs_tight):
            bbox = extend_bbox(bbox_tight.copy(),4.0*np.max(radius),g_resolution_y/g_resolution_x,I.shape)
            im_crop = crop_resize(I, bbox, (g_resolution_x, g_resolution_y))
            bgr_crop = crop_resize(B, bbox, (g_resolution_x, g_resolution_y))
            input_batch = torch.cat((transforms.ToTensor()(im_crop), transforms.ToTensor()(bgr_crop)), 0).unsqueeze(0).float()
            with torch.no_grad():
                renders = self.defmo(input_batch.to(self.device))
            renders_rgba = renders[0].data.cpu().detach().numpy().transpose(2,3,1,0)
            est_hs = rev_crop_resize(renders_rgba,bbox,np.zeros((I.shape[0],I.shape[1],4)))
            im_crop = crop_resize(I, self.bbox, new_res)
            self.bgr_crop = crop_resize(B, self.bbox, new_res)
            input_batch = torch.cat((transforms.ToTensor()(im_crop), transforms.ToTensor()(self.bgr_crop)), 0).unsqueeze(0).float()
            defmo_masks = crop_resize(est_hs, self.bbox, new_res)
            hs_frames = torch.zeros((1,nsplits*self.config["factor"],4,input_batch.shape[-2],input_batch.shape[-1]))
            for tti in range(nsplits*self.config["factor"]):
                hs_frames[0,tti] = transforms.ToTensor()(defmo_masks[:,:,:,tti])
            input_batch_sfb = torch.cat( (input_batch_sfb, input_batch), 0)
            hs_frames_sfb = torch.cat( (hs_frames_sfb, hs_frames), 0)
        hs_frames_sfb = sync_directions_rgba(hs_frames_sfb)
        best_model = self.apply_mfb(input_batch_sfb, hs_frames_sfb)
        best_model["renders"] = best_model["renders"].reshape(1,self.config["input_frames"],nsplits,self.config["factor"],4,best_model["renders"].shape[-2],-1).mean(3)
        return best_model

    def apply_mfb(self, input_batch, hs_frames):
        input_batch, hs_frames = input_batch[None].to(self.device), hs_frames[None].to(self.device)
        config = self.config.copy()
        if hs_frames[:,:,:,3:4].max() < 0.1:
            hs_frames[:,:,:,3:4] = torch.ones(hs_frames[:,:,:,3:4].shape)
        width = hs_frames.shape[-1]
        height = hs_frames.shape[-2]
        best_model = {}
        best_model["value"] = 100
        preoptimization = True
        for prot in config["shapes"]: 
            if prot == 'sphere':
                ivertices, faces, iface_features = generate_initial_mesh(config["mesh_size"])
            else:
                mesh = load_obj(os.path.join('.','prototypes',prot+'.obj'))
                ivertices = mesh.vertices.numpy()
                faces = mesh.faces.numpy().copy()
                iface_features = mesh.uvs[mesh.face_uvs_idx].numpy()

            torch.backends.cudnn.benchmark = True
            loss_function = FMOLoss(config, ivertices, faces).to(self.device)

            for predict_vertices in config["predict_vertices_list"]:
                config["erode_renderer_mask"] = self.config["erode_renderer_mask"]
                config["predict_vertices"] = predict_vertices
                config["loss_laplacian_weight"] = 3*self.config["loss_laplacian_weight"] # for pre-optimization

                rendering = RenderingKaolinMulti(config, faces, width, height).to(self.device)
                encoder = EncoderMulti(config, ivertices, faces, iface_features, width, height).to(self.device)
                all_parameters = list(encoder.parameters())
                optimizer = torch.optim.Adam(all_parameters, lr = config["learning_rate"])
                encoder.train()
                for epoch in range(config["iterations"]):
                    obj = encoder()
                    renders = rendering(obj)
                    sloss, mloss, lap_loss, loss_tv, jloss, model_loss_perframe = loss_function(renders, hs_frames, input_batch, obj["translation"], obj["quaternion"], obj["vertices"], obj["texture_maps"], rendering.faces)

                    jloss = jloss.mean()
                    optimizer.zero_grad()
                    jloss.backward()
                    optimizer.step()
                    model_loss = mloss.mean().item()
                    if config["verbose"] and epoch % 20 == 0:
                        print("Epoch {:4d}".format(epoch+1), end =" ")
                        if config["loss_use_supervised"]:
                            print(", loss {:.3f}".format(sloss.mean().item()), end =" ")
                        if config["loss_use_model"]:
                            print(", model {:.3f}".format(model_loss), end =" ")
                        if config["loss_laplacian_weight"] > 0:
                            print(", lap {:.3f}".format(lap_loss.mean().item()), end =" ")
                        if config["loss_total_variation"] > 0:
                            print(", TV {:.3f}".format((loss_tv.mean().item())), end =" ")
                        print(", joint {:.3f}".format(jloss.item()))
                    
                    if preoptimization and (epoch >= 100 or best_model["value"] < 0.3):
                        preoptimization = False
                        config["loss_use_model"] = True
                        config["erode_renderer_mask"] = 3
                    if epoch >= 300: # for finer convergence
                        config["erode_renderer_mask"] = 5
                        config["loss_laplacian_weight"] = self.config["loss_laplacian_weight"]
                    if model_loss < best_model["value"]:
                        best_model["value"] = model_loss
                        best_model["value_joint"] = model_loss_perframe
                        best_model["renders"] = renders.detach().cpu().numpy()
                        best_model["vertices"] = obj["vertices"].detach().clone()
                        best_model["texture_maps"] = obj["texture_maps"].detach().clone()
                        best_model["translation"] = obj["translation"].detach().clone()
                        best_model["quaternion"] = obj["quaternion"].detach().clone()
                        best_model["face_features"] = obj["face_features"].detach().clone()
                        best_model["exp"] = obj["exp"].detach().clone()
                        best_model["faces"] = faces
                        best_model["prototype"] = prot
                        best_model["predict_vertices"] = predict_vertices
        return best_model


def sync_directions_rgba(est_hs):
    for frmi in range(1,est_hs.shape[0]):
        tsr0 = est_hs[frmi-1]
        tsr = est_hs[frmi]
        if frmi == 1:
            forward = np.min([torch.mean((tsr0[-1] - tsr[-1])**2), torch.mean((tsr0[-1] - tsr[0])**2)])
            backward = np.min([torch.mean((tsr0[0] - tsr[-1])**2), torch.mean((tsr0[0] - tsr[0])**2)])
            if backward < forward:
                est_hs[frmi-1] = torch.flip(est_hs[frmi-1],[0])
                tsr0 = est_hs[frmi-1]

        if torch.mean((tsr0[-1] - tsr[-1])**2) < torch.mean((tsr0[-1] - tsr[0])**2):
            ## reverse time direction for better alignment
            est_hs[frmi] = torch.flip(est_hs[frmi],[0])
    return est_hs