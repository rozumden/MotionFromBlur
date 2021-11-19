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

class ShapeFromBlur():
    def __init__(self, config = None, device = None):
        if config is None:
            config = load_config("configs/config_optimize.yaml")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.device = device
        g_saved_models_folder = '/cluster/home/denysr/src/defmo/saved_models/'
        self.encoder = EncoderCNN().to(device)
        self.rendering = RenderingCNN().to(device)
        self.encoder.load_state_dict(torch.load(os.path.join(g_saved_models_folder, 'encoder_best.pt'),map_location=device))
        self.rendering.load_state_dict(torch.load(os.path.join(g_saved_models_folder, 'rendering_best.pt'),map_location=device))
        self.encoder.train(False)
        self.rendering.train(False)

    def apply(self,Is,Bs,bboxs_tight,nsplits,radius,obj_dim):
        input_batch_sfb = torch.Tensor([])
        hs_frames_sfb = torch.Tensor([])
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
            
            self.bbox = extend_bbox(bbox_tight.copy(),1.0*np.max(radius),g_resolution_y/g_resolution_x,I.shape)
            im_crop = crop_resize(I, self.bbox, (g_resolution_x, g_resolution_y))
            self.bgr_crop = crop_resize(B, self.bbox, (g_resolution_x, g_resolution_y))
            input_batch = torch.cat((transforms.ToTensor()(im_crop), transforms.ToTensor()(self.bgr_crop)), 0).unsqueeze(0).float()
            defmo_masks = crop_resize(est_hs, self.bbox, (g_resolution_x, g_resolution_y))
            hs_frames = torch.zeros((1,nsplits*self.config["factor"],4,input_batch.shape[-2],input_batch.shape[-1]))
            for tti in range(nsplits*self.config["factor"]):
                hs_frames[0,tti] = transforms.ToTensor()(defmo_masks[:,:,:,tti])
            input_batch_sfb = torch.cat( (input_batch_sfb, input_batch), 0)
            hs_frames_sfb = torch.cat( (hs_frames_sfb, hs_frames), 0)
        best_model = self.apply_sfb(input_batch_sfb, hs_frames_sfb)
        if "hs_frames" in best_model:
            best_model["hs_frames"] = best_model["hs_frames"].reshape(1,self.config["input_frames"],nsplits,self.config["factor"],4,renders.shape[-2],-1).mean(3)
        best_model["renders"] = best_model["renders"].reshape(1,self.config["input_frames"],nsplits,self.config["factor"],4,renders.shape[-2],-1).mean(3)
        return best_model

    def apply_sfb(self, input_batch, hs_frames):
        input_batch, hs_frames = input_batch[None].to(self.device), hs_frames[None].to(self.device)
        config = self.config.copy()
        # hs_frames[:,:,:,3:4][hs_frames[:,:,:,3:4] < 0.9] = 0
        # hs_frames[:,:,:,3:4][hs_frames[:,:,:,:3].mean(3)[:,:,:,None] < 0.25] = 0
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
        for prot in config["shapes"]: 
            if prot == 'sphere':
                ivertices, faces, iface_features = generate_initial_mesh(config["mesh_size"])
            else:
                mesh = load_obj(os.path.join('/cluster/home/denysr/src/ShapeFromBlur/prototypes',prot+'.obj'))
                ivertices = mesh.vertices.numpy()
                faces = mesh.faces.numpy().copy()
                iface_features = mesh.uvs[mesh.face_uvs_idx].numpy()

            torch.backends.cudnn.benchmark = True
            rendering = RenderingKaolin(config, faces, width, height).to(self.device)
            loss_function = FMOLoss(config, ivertices, faces).to(self.device)

            for predict_vertices in config["predict_vertices_list"]:
                config["erode_renderer_mask"] = self.config["erode_renderer_mask"]
                config["loss_iou_weight"] = init_vals[0]
                config["loss_rgb_weight"] = init_vals[1]
                config["sigmainv"] = self.config["sigmainv"]

                config["predict_vertices"] = predict_vertices
                encoder = EncoderBasic(config, ivertices, faces, iface_features, width, height).to(self.device)
                breakpoint()
                
               	if config["verbose"]:
                    print('Total params {}'.format(sum(p.numel() for p in encoder.parameters())))

                all_parameters = list(encoder.parameters())
                optimizer = torch.optim.Adam(all_parameters, lr = config["learning_rate"])

                encoder.train()
                for epoch in range(config["iterations"]):
                    translation, quaternion, vertices, face_features, texture_maps = encoder()
                    renders = rendering(translation, quaternion, vertices, face_features, texture_maps)
                    sloss, mloss, ael_loss, lap_loss, normal_loss, loss_tv, jloss = loss_function(renders, hs_frames, input_batch, translation, quaternion, vertices, texture_maps, rendering.faces)

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
                    
                    if epoch == 99:
                        config["erode_renderer_mask"] = 5
                        config["loss_rgb_weight"] /= 2
                    elif epoch == 199:
                        config["erode_renderer_mask"] = 7
                        config["loss_rgb_weight"] = 0
                        # config["loss_iou_weight"] = 0
                    elif epoch == 299:
                        config["erode_renderer_mask"] = 11
                        # config["sigmainv"] = 70000

                    if model_loss < best_model["value"]:
                        best_model["value"] = model_loss
                        best_model["renders"] = renders.detach().cpu().numpy()

                        if config["write_results"]:
                            best_model["vertices"] = vertices.detach().clone()
                            best_model["texture_maps"] = texture_maps.detach().clone()
                            best_model["translation"] = translation.detach().clone()
                            best_model["quaternion"] = quaternion.detach().clone()
                            best_model["face_features"] = face_features.detach().clone()
                            best_model["faces"] = faces
                            best_model["prototype"] = prot
                            best_model["predict_vertices"] = predict_vertices
                            write_renders(renders, input_batch, hs_frames, config, tmp_folder)
                            save_image(torch.nn.UpsamplingBilinear2d(scale_factor=4)(best_model["texture_maps"]), os.path.join(tmp_folder+'paper_imgs/','tex.png'))
                            
        # config["erode_renderer_mask"] = 9
        # rendering = RenderingKaolin(config, best_model["faces"], width, height).to(self.device)
        # renders = rendering(best_model["translation"], best_model["quaternion"], best_model["vertices"], best_model["face_features"], best_model["texture_maps"])
        # best_model["renders"] = renders.detach().cpu().numpy()

        if config["write_results"]:
            write_renders(renders, input_batch, hs_frames, config, tmp_folder)
            write_obj_mesh(best_model["vertices"][0].cpu().numpy(), best_model["faces"], best_model["face_features"][0].cpu().numpy(), os.path.join(tmp_folder+'paper_imgs/','mesh.obj'))
            save_image(best_model["texture_maps"], os.path.join(tmp_folder+'paper_imgs/','tex.png'))
            print("Best model type {}, predict vertices {}".format(best_model["prototype"],best_model["predict_vertices"]))
            best_model["hs_frames"] = hs_frames.detach().cpu().numpy()

        if config["apply_blur_inside"] > 0:
            for ki in range(best_model["renders"].shape[2]): 
                best_model["renders"][0,0,ki,3] = gaussian_filter(best_model["renders"][0,0,ki,3], sigma=3*config["apply_blur_inside"])

        if config["mask_iou_th"] > 0:
            if calciou_masks(best_model["renders"][0,0,0,3,:,:], best_model["renders"][0,0,-1,3,:,:]) > config["mask_iou_th"]:
                best_model["renders"] = np.repeat(input_batch.detach().cpu().numpy()[:,:,None,:4],config["fmo_steps"],2)
                best_model["renders"][:,:,:,3] = 1.0

        return best_model


    def finetune_sfb(self, input_batch, hs_frames, best_model):
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

        faces = best_model["faces"]
        ivertices = best_model["vertices"].to(self.device)
        iface_features = best_model["face_features"].to(self.device)

        torch.backends.cudnn.benchmark = True
        rendering = RenderingKaolin(config, faces, width, height).to(self.device)
        loss_function = FMOLoss(config, ivertices, faces).to(self.device)

        config["erode_renderer_mask"] = self.config["erode_renderer_mask"]
        config["loss_iou_weight"] = init_vals[0]
        config["loss_rgb_weight"] = init_vals[1]
        config["sigmainv"] = self.config["sigmainv"]
        
        predict_vertices = best_model["predict_vertices"]
        config["predict_vertices"] = predict_vertices

        encoder = EncoderBasic(config, ivertices, faces, iface_features, width, height).to(self.device)

        if config["verbose"]:
            print('Total params {}'.format(sum(p.numel() for p in encoder.parameters())))

        all_parameters = list(encoder.parameters())
        optimizer = torch.optim.Adam(all_parameters, lr = config["learning_rate"])

        encoder.train()
        for epoch in range(config["finetune_iterations"]):
            translation, quaternion, vertices, face_features, texture_maps = encoder()
            renders = rendering(translation, quaternion, vertices, face_features, texture_maps)
            sloss, mloss, ael_loss, lap_loss, normal_loss, loss_tv, jloss = loss_function(renders, hs_frames, input_batch, translation, quaternion, vertices, texture_maps, rendering.faces)

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
            
            if epoch == 99:
                config["erode_renderer_mask"] = 5
                config["loss_rgb_weight"] /= 2
            elif epoch == 199:
                config["erode_renderer_mask"] = 7
                config["loss_rgb_weight"] = 0
                # config["loss_iou_weight"] = 0
            elif epoch == 299:
                config["erode_renderer_mask"] = 11
                # config["sigmainv"] = 70000

            if model_loss < best_model["value"]:
                best_model["value"] = model_loss
                best_model["renders"] = renders.detach().cpu().numpy()

                if config["write_results"]:
                    best_model["vertices"] = vertices.detach().clone()
                    best_model["texture_maps"] = texture_maps.detach().clone()
                    best_model["translation"] = translation.detach().clone()
                    best_model["quaternion"] = quaternion.detach().clone()
                    best_model["face_features"] = face_features.detach().clone()
                    best_model["faces"] = faces
                    best_model["prototype"] = prot
                    best_model["predict_vertices"] = predict_vertices
                    write_renders(renders, input_batch, hs_frames, config, tmp_folder)
                    save_image(torch.nn.UpsamplingBilinear2d(scale_factor=4)(best_model["texture_maps"]), os.path.join(tmp_folder+'paper_imgs/','tex.png'))
                            
        if config["write_results"]:
            write_renders(renders, input_batch, hs_frames, config, tmp_folder)
            write_obj_mesh(best_model["vertices"][0].cpu().numpy(), best_model["faces"], best_model["face_features"][0].cpu().numpy(), os.path.join(tmp_folder+'paper_imgs/','mesh.obj'))
            save_image(best_model["texture_maps"], os.path.join(tmp_folder+'paper_imgs/','tex.png'))
            print("Best model type {}, predict vertices {}".format(best_model["prototype"],best_model["predict_vertices"]))
            best_model["hs_frames"] = hs_frames.detach().cpu().numpy()

        if config["apply_blur_inside"] > 0:
            for ki in range(best_model["renders"].shape[2]): 
                best_model["renders"][0,0,ki,3] = gaussian_filter(best_model["renders"][0,0,ki,3], sigma=3*config["apply_blur_inside"])

        if config["mask_iou_th"] > 0:
            if calciou_masks(best_model["renders"][0,0,0,3,:,:], best_model["renders"][0,0,-1,3,:,:]) > config["mask_iou_th"]:
                best_model["renders"] = np.repeat(input_batch.detach().cpu().numpy()[:,:,None,:4],config["fmo_steps"],2)
                best_model["renders"][:,:,:,3] = 1.0

        return best_model


