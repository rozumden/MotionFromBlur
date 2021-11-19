import os
import torch
import numpy as np

from utils import *
from models.initial_mesh import generate_initial_mesh
from models.kaolin_wrapper import load_obj, write_obj_mesh

from models.encoder import *
from models.rendering import *
from models.loss import *

from DeFMO.models.encoder import EncoderCNN
from DeFMO.models.rendering import RenderingCNN
from torchvision import transforms

from helpers.torch_helpers import write_renders
from scipy.ndimage.filters import gaussian_filter

class MotionFromBlur():
    def __init__(self, config = None, device = None):
        if config is None:
            config = load_config("configs/config_optimize.yaml")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.device = device
        # TODO: paste DeFMO path
        g_saved_models_folder = '/cluster/home/denysr/src/defmo/saved_models/'
        self.encoder = EncoderCNN().to(device)
        self.rendering = RenderingCNN().to(device)
        self.encoder.load_state_dict(torch.load(os.path.join(g_saved_models_folder, 'encoder_best.pt'),map_location=device))
        self.rendering.load_state_dict(torch.load(os.path.join(g_saved_models_folder, 'rendering_best.pt'),map_location=device))
        self.encoder.train(False)
        self.rendering.train(False)

    def apply(self,Is,Bs,bboxs_tight,nsplits,radius,obj_dim):
        self.config["input_frames"] = len(Is)
        self.bbox = bboxs_tight[0].copy()
        for bbox_tight in bboxs_tight:
            bbox = extend_bbox(bbox_tight.copy(),1.0*np.max(radius),g_resolution_y/g_resolution_x,Is[0].shape)
            bbox = bbox_tight.copy()
            self.bbox[:2] = np.c_[self.bbox[:2], bbox[:2]].min(1)
            self.bbox[2:] = np.c_[self.bbox[2:], bbox[2:]].max(1)
        self.bbox = extend_bbox_uniform(self.bbox.copy(),0.5*np.max(radius),Is[0].shape)
        input_batch_sfb = torch.Tensor([])
        hs_frames_sfb = torch.Tensor([])
        new_res = ((self.bbox[3]-self.bbox[1]), (self.bbox[2]-self.bbox[0]))
        for I, B, bbox_tight in zip(Is,Bs,bboxs_tight):
            bbox = extend_bbox(bbox_tight.copy(),4.0*np.max(radius),g_resolution_y/g_resolution_x,I.shape)
            im_crop = crop_resize(I, bbox, (g_resolution_x, g_resolution_y))
            bgr_crop = crop_resize(B, bbox, (g_resolution_x, g_resolution_y))
            input_batch = torch.cat((transforms.ToTensor()(im_crop), transforms.ToTensor()(bgr_crop)), 0).unsqueeze(0).float()
            with torch.no_grad():
                latent = self.encoder(input_batch.to(self.device))
                renders = self.rendering(latent,torch.linspace(0.01,0.99,nsplits*self.config["factor"]).to(self.device)[None].repeat(1,1))

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
        if "hs_frames" in best_model:
            best_model["hs_frames"] = best_model["hs_frames"].reshape(1,self.config["input_frames"],nsplits,self.config["factor"],4,best_model["hs_frames"].shape[-2],-1).mean(3)
        best_model["renders"] = best_model["renders"].reshape(1,self.config["input_frames"],nsplits,self.config["factor"],4,best_model["renders"].shape[-2],-1).mean(3)
        return best_model

    def apply_mfb(self, input_batch, hs_frames):
        input_batch, hs_frames = input_batch[None].to(self.device), hs_frames[None].to(self.device)
        config = self.config.copy()
        if hs_frames[:,:,:,3:4].max() < 0.1:
            init_vals = [0.01, 0.0]
            hs_frames[:,:,:,3:4] = torch.ones(hs_frames[:,:,:,3:4].shape)
        else:
            init_vals = [self.config["loss_iou_weight"], self.config["loss_rgb_weight"]]

        config["fmo_steps"] = hs_frames.shape[2]
        if config["write_results"]:
            save_image(input_batch[0,:,:3],os.path.join(tmp_folder+'paper_imgs/','im.png'))
            save_image(hs_frames[0].view(config["input_frames"]*config["fmo_steps"],4,hs_frames.shape[-2],-1),os.path.join(tmp_folder+'paper_imgs/','renders_hs.png'))

        width = hs_frames.shape[-1]
        height = hs_frames.shape[-2]
        best_model = {}
        best_model["value"] = 100
        best_model["phase"] = 0
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
                config["loss_iou_weight"] = init_vals[0]
                config["loss_rgb_weight"] = init_vals[1]
                config["sigmainv"] = self.config["sigmainv"]

                config["predict_vertices"] = predict_vertices

                if config["number_of_pieces"] > 0:
                    rendering = RenderingKaolinMulti(config, faces, width, height).to(self.device)
                    encoder = EncoderMulti(config, ivertices, faces, iface_features, width, height).to(self.device)
                else:
                    rendering = RenderingKaolin(config, faces, width, height).to(self.device)
                    encoder = EncoderBasic(config, ivertices, faces, iface_features, width, height).to(self.device)

               	if config["verbose"]:
                    print('Total params {}'.format(sum(p.numel() for p in encoder.parameters())))

                all_parameters = list(encoder.parameters())
                optimizer = torch.optim.Adam(all_parameters, lr = config["learning_rate"])

                encoder.train()
                for epoch in range(config["iterations"]):
                    obj = encoder()
                    renders = rendering(obj)
                    sloss, mloss, ael_loss, lap_loss, normal_loss, loss_tv, jloss, model_loss_perframe = loss_function(renders, hs_frames, input_batch, obj["translation"], obj["quaternion"], obj["vertices"], obj["texture_maps"], rendering.faces)

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
                        if config["loss_ael_weight"] > 0:
                            print(", ael {:.3f}".format(ael_loss.mean().item()), end =" ")
                        if config["loss_laplacian_weight"] > 0:
                            print(", lap {:.3f}".format(lap_loss.mean().item()), end =" ")
                        if config["loss_normal_weight"] > 0:
                            print(", normal {:.3f}".format((normal_loss.mean().item())), end =" ")
                        if config["loss_total_variation"] > 0:
                            print(", TV {:.3f}".format((loss_tv.mean().item())), end =" ")
                        print(", joint {:.3f}".format(jloss.item()))
                    
                    if best_model["phase"] == 0:
                        if epoch >= 100 or best_model["value"] < 0.3:
                            best_model["phase"] = 1
                            best_model["value"] = 100
                            config["loss_use_model"] = True
                            config["erode_renderer_mask"] = 3
                    elif best_model["phase"] == 1:
                        if epoch >= 300 or best_model["value"] < 0.05:
                            config["erode_renderer_mask"] = 5
                            best_model["phase"] = 2

                    if model_loss < best_model["value"]:
                        best_model["value"] = model_loss
                        best_model["value_joint"] = model_loss_perframe
                        best_model["renders"] = renders.detach().cpu().numpy()
                        if config["write_results"] or config["finetune_iterations"] > 0:
                            best_model["vertices"] = obj["vertices"].detach().clone()
                            best_model["texture_maps"] = obj["texture_maps"].detach().clone()
                            best_model["translation"] = obj["translation"].detach().clone()
                            best_model["quaternion"] = obj["quaternion"].detach().clone()
                            best_model["face_features"] = obj["face_features"].detach().clone()
                            best_model["exp"] = obj["exp"].detach().clone()
                            best_model["faces"] = faces
                            best_model["prototype"] = prot
                            best_model["predict_vertices"] = predict_vertices
                        if config["finetune_iterations"] > 0:
                            best_model["enc_text"] = encoder.texture_map
                            best_model["enc_translation"] = encoder.translation
                            best_model["enc_quaternion"] = encoder.quaternion
                            best_model["enc_exp"] = encoder.exposure_fraction
                            if best_model["predict_vertices"]:
                                best_model["enc_ver"] = encoder.vertices
                            best_model["iface_features"] = iface_features
                            best_model["ivertices"] = ivertices
                        if config["write_results"]:
                            write_renders(renders, input_batch, hs_frames, config, tmp_folder)
                            save_image(torch.nn.UpsamplingBilinear2d(scale_factor=4)(best_model["texture_maps"]), os.path.join(tmp_folder+'paper_imgs/','tex.png'))
                            
        if config["write_results"]:
            write_renders(renders, input_batch, hs_frames, config, tmp_folder)
            write_obj_mesh(best_model["vertices"][0].cpu().numpy(), best_model["faces"], best_model["face_features"][0].cpu().numpy(), os.path.join(tmp_folder+'paper_imgs/','mesh.obj'))
            save_image(best_model["texture_maps"], os.path.join(tmp_folder+'paper_imgs/','tex.png'))
            print("Best model type {}, predict vertices {}".format(best_model["prototype"],best_model["predict_vertices"]))
            best_model["hs_frames"] = hs_frames.detach().cpu().numpy()

        return best_model


