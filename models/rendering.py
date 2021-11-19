import torch
import torch.nn as nn
from main_settings import *
import kaolin
from models.kaolin_wrapper import *
from kornia.geometry.conversions import angle_axis_to_rotation_matrix, quaternion_to_angle_axis, rotation_matrix_to_quaternion
from kornia.geometry.conversions import quaternion_to_rotation_matrix, rotation_matrix_to_angle_axis
from kornia.morphology import erosion, dilation
from kornia.filters import GaussianBlur2d
from utils import *

def mesh_rotate(best_model,config, frmi_needed, ti_needed):
    config["fmo_steps"] = best_model["hs_frames"].shape[2]
    sizes = (best_model["translation"][:,:,:,1]**2).sum((0,1,3)).sqrt() + 1e-10
    prop = sizes / sizes.sum()
    prop = (prop * config["fmo_steps"] * config["input_frames"]).round().to(int)
    prop[0] = config["fmo_steps"] * config["input_frames"] - prop[1:].sum()
    unit_vertices = best_model["vertices"]
    rotation_matrix = angle_axis_to_rotation_matrix(quaternion_to_angle_axis(best_model["quaternion"][:,0,0]))    
    translation_start = best_model["translation"][:,:,0,0]    
    frames_in_piece = 0
    time_before = 0
    current_frmi = 0
    for pind in range(config["number_of_pieces"]):
        angles_per_frame = quaternion_to_angle_axis(best_model["quaternion"][:,pind,1])/config["rotation_divide"]
        rotation_matrix_step = angle_axis_to_rotation_matrix(best_model["exp"]*angles_per_frame/config["fmo_steps"])
        rotation_matrix_noexp = angle_axis_to_rotation_matrix(angles_per_frame*(1-best_model["exp"]))

        for ti in torch.linspace(0,1,prop[pind]):
            if frames_in_piece == 0:
                time_before = ti
            time_exp = time_before + (ti-time_before)*best_model["exp"]
            
            if current_frmi == frmi_needed and ti_needed == frames_in_piece:
                break

            frames_in_piece += 1
            if frames_in_piece == config["fmo_steps"]:
                current_frmi += 1
                frames_in_piece = 0
                rotation_matrix = torch.matmul(rotation_matrix, rotation_matrix_noexp)
            else:
                rotation_matrix = torch.matmul(rotation_matrix, rotation_matrix_step)
        if current_frmi == frmi_needed and ti_needed == frames_in_piece:
            break
        for power_ind in range(1,config["number_of_terms"]): 
            translation_start = translation_start + best_model["translation"][:,:,pind,power_ind]

    vertices = kaolin.render.camera.rotate_translate_points(unit_vertices, rotation_matrix, torch.zeros((1,3)).to(unit_vertices.device)) + translation_start
    for power_ind in range(1,config["number_of_terms"]): 
        vertices = vertices + (time_exp**power_ind)*best_model["translation"][:,:,pind,power_ind]
    return vertices   

def get_3dtraj(best_model,config):
    config["fmo_steps"] = best_model["hs_frames"].shape[2]
    sizes = (best_model["translation"][:,:,:,1]**2).sum((0,1,3)).sqrt() + 1e-10
    prop = sizes / sizes.sum()
    prop = (prop * config["fmo_steps"] * config["input_frames"]).round().to(int)
    prop[0] = config["fmo_steps"] * config["input_frames"] - prop[1:].sum()
    unit_vertices = best_model["vertices"]
    translation_start = best_model["translation"][:,:,0,0]    
    frames_in_piece = 0
    time_before = 0
    current_frmi = 0
    points = []
    for pind in range(config["number_of_pieces"]):
        for ti in torch.linspace(0,1,prop[pind]):
            if frames_in_piece == 0:
                time_before = ti
            time_exp = time_before + (ti-time_before)*best_model["exp"]

            if frames_in_piece == config["fmo_steps"]-1 or frames_in_piece == 0:
                trans = translation_start.clone()
                for power_ind in range(1,config["number_of_terms"]): 
                    trans = trans + (time_exp**power_ind)*best_model["translation"][:,:,pind,power_ind]
                points.append(trans[0,0].cpu().numpy())

            frames_in_piece += 1
            if frames_in_piece == config["fmo_steps"]:
                current_frmi += 1
                frames_in_piece = 0

        for power_ind in range(1,config["number_of_terms"]): 
            translation_start = translation_start + best_model["translation"][:,:,pind,power_ind]

    return points     

class RenderingKaolinMulti(nn.Module):
    def __init__(self, config, faces, width, height):
        super().__init__()
        self.config = config
        self.height = height
        self.width = width
        camera_proj = kaolin.render.camera.generate_perspective_projection(1.57/2, self.width/self.height ) # 45 degrees
        self.register_buffer('camera_proj', camera_proj)
        self.register_buffer('camera_trans', torch.Tensor([0,0,self.config["camera_distance"]])[None])
        self.register_buffer('obj_center', torch.zeros((1,3)))
        camera_up_direction = torch.Tensor((0,1,0))[None]
        camera_rot,_ = kaolin.render.camera.generate_rotate_translate_matrices(self.camera_trans, self.obj_center, camera_up_direction)
        self.register_buffer('camera_rot', camera_rot)
        self.set_faces(faces)
            
    def set_faces(self, faces):
        self.register_buffer('faces', torch.LongTensor(faces))

    def forward(self, obj):
        sizes = (obj["translation"][:,:,:,1]**2).sum((0,1,3)).sqrt() + 1e-10
        prop = sizes / sizes.sum()
        prop = (prop * self.config["fmo_steps"] * self.config["input_frames"]).round().to(int)
        prop[0] = self.config["fmo_steps"] * self.config["input_frames"] - prop[1:].sum()

        unit_vertices = obj["vertices"]
        kernel = torch.ones(self.config["erode_renderer_mask"], self.config["erode_renderer_mask"]).to(obj["translation"].device)
        all_renders = []
        renders = []
        frames_in_piece = 0
        time_before = 0
        rotation_matrix = angle_axis_to_rotation_matrix(quaternion_to_angle_axis(obj["quaternion"][:,0,0]))    
        translation_start = obj["translation"][:,:,0,0]
        # breakpoint()
        for pind in range(self.config["number_of_pieces"]):
            angles_per_frame = quaternion_to_angle_axis(obj["quaternion"][:,pind,1])/self.config["rotation_divide"]
            rotation_matrix_step = angle_axis_to_rotation_matrix(obj["exp"]*angles_per_frame/self.config["fmo_steps"])
            rotation_matrix_noexp = angle_axis_to_rotation_matrix(angles_per_frame*(1-obj["exp"]))

            for ti in torch.linspace(0,1,prop[pind]):
                if frames_in_piece == 0:
                    time_before = ti
                time_exp = time_before + (ti-time_before)*obj["exp"]
                
                vertices = kaolin.render.camera.rotate_translate_points(unit_vertices, rotation_matrix, self.obj_center) + translation_start
                for power_ind in range(1,self.config["number_of_terms"]): 
                    vertices = vertices + (time_exp**power_ind)*obj["translation"][:,:,pind,power_ind]
                
                face_vertices_cam, face_vertices_image, face_normals = prepare_vertices(vertices, self.faces, self.camera_rot, self.camera_trans, self.camera_proj)
                face_vertices_z = face_vertices_cam[:,:,:,-1]
                face_normals_z = face_normals[:,:,-1]
                
                ren_features, ren_mask, red_index = kaolin.render.mesh.dibr_rasterization(self.height, self.width, face_vertices_z, face_vertices_image, obj["face_features"], face_normals_z, sigmainv=self.config["sigmainv"], boxlen=0.02, knum=30, multiplier=1000)
                if not obj["texture_maps"] is None:
                    ren_features = kaolin.render.mesh.texture_mapping(ren_features, obj["texture_maps"], mode='bilinear')
                result = ren_features.permute(0,3,1,2)
                if self.config["erode_renderer_mask"] > 0:
                    ren_mask = erosion(ren_mask[:,None], kernel)[:,0]
                if self.config["apply_blur_inside"] > 0:
                    gauss = GaussianBlur2d((11, 11), (self.config["apply_blur_inside"], self.config["apply_blur_inside"]))
                    result_rgba = torch.cat((gauss(result),gauss(gauss(ren_mask[:,None].float()))),1)
                else:
                    result_rgba = torch.cat((result,ren_mask[:,None]),1)
                renders.append(result_rgba)

                frames_in_piece += 1
                if frames_in_piece == self.config["fmo_steps"]:
                    frames_in_piece = 0
                    renders = torch.stack(renders,1).contiguous()
                    all_renders.append(renders)
                    renders = []
                    rotation_matrix = torch.matmul(rotation_matrix, rotation_matrix_noexp)
                else:
                    rotation_matrix = torch.matmul(rotation_matrix, rotation_matrix_step)

            for power_ind in range(1,self.config["number_of_terms"]): 
                translation_start = translation_start + obj["translation"][:,:,pind,power_ind]

        all_renders = torch.stack(all_renders,1).contiguous()
        return all_renders

class RenderingKaolin(nn.Module):
    def __init__(self, config, faces, width, height):
        super().__init__()
        self.config = config
        self.height = height
        self.width = width
        camera_proj = kaolin.render.camera.generate_perspective_projection(1.57/2, self.width/self.height ) # 45 degrees
        self.register_buffer('camera_proj', camera_proj)
        self.register_buffer('camera_trans', torch.Tensor([0,0,self.config["camera_distance"]])[None])
        self.register_buffer('obj_center', torch.zeros((1,3)))
        camera_up_direction = torch.Tensor((0,1,0))[None]
        camera_rot,_ = kaolin.render.camera.generate_rotate_translate_matrices(self.camera_trans, self.obj_center, camera_up_direction)
        self.register_buffer('camera_rot', camera_rot)
        self.set_faces(faces)
            
    def set_faces(self, faces):
        self.register_buffer('faces', torch.LongTensor(faces))

    def forward(self, obj):
        unit_vertices = obj["vertices"]
        kernel = torch.ones(self.config["erode_renderer_mask"], self.config["erode_renderer_mask"]).to(obj["translation"].device)
        
        all_renders = []
        for frmi in range(obj["quaternion"].shape[1]):
            if frmi == 0 or not self.config["connect_frames"]: 
                rotation_matrix = angle_axis_to_rotation_matrix(quaternion_to_angle_axis(obj["quaternion"][:,frmi,1]))    
                rotation_matrix_noexp = angle_axis_to_rotation_matrix(obj["exp"]*quaternion_to_angle_axis(obj["quaternion"][:,frmi,0])/self.config["rotation_divide"])
                rotation_matrix_step = angle_axis_to_rotation_matrix(quaternion_to_angle_axis(obj["quaternion"][:,frmi,0])/self.config["fmo_steps"]/self.config["rotation_divide"])
            else:
                if self.config["separate_rotations"]:
                    rotation_matrix_step = angle_axis_to_rotation_matrix(quaternion_to_angle_axis(obj["quaternion"][:,frmi,0])/self.config["fmo_steps"]/self.config["rotation_divide"])
                if self.config["apply_exposure_fraction"]:
                    rotation_matrix = torch.matmul(rotation_matrix, rotation_matrix_noexp)

            if frmi == 0 or self.config["separate_translations"]:
                translation_start = obj["translation"][:,:,frmi,1]
                translation_step = obj["translation"][:,:,frmi,0]
            else:
                translation_start = translation_start + translation_step

            renders = []
            for ti in torch.linspace(0,1,self.config["fmo_steps"]):
                vertices = kaolin.render.camera.rotate_translate_points(unit_vertices, rotation_matrix, self.obj_center) 
                vertices = vertices + translation_start + ti*translation_step

                face_vertices_cam, face_vertices_image, face_normals = prepare_vertices(vertices, self.faces, self.camera_rot, self.camera_trans, self.camera_proj)
                face_vertices_z = face_vertices_cam[:,:,:,-1]
                face_normals_z = face_normals[:,:,-1]
                ren_features, ren_mask, red_index = kaolin.render.mesh.dibr_rasterization(self.height, self.width, face_vertices_z, face_vertices_image, obj["face_features"], face_normals_z, sigmainv=self.config["sigmainv"], boxlen=0.02, knum=30, multiplier=1000)
                if not obj["texture_maps"] is None:
                    ren_features = kaolin.render.mesh.texture_mapping(ren_features, obj["texture_maps"], mode='bilinear')
                result = ren_features.permute(0,3,1,2)
                if self.config["erode_renderer_mask"] > 0:
                    ren_mask = erosion(ren_mask[:,None], kernel)[:,0]
                    # kernel_rgb = torch.ones(5, 5).to(translation.device)
                    # ren_feat_dil = dilation(result, kernel_rgb)
                    # masked = 1.0*(red_index == -1)[:,None][:,[0,0,0]]
                    # result = ren_feat_dil*masked + result*(1-masked)
                # hard_mask = 1.0*(red_index > -1)[:,None]
                if self.config["apply_blur_inside"] > 0:
                    gauss = GaussianBlur2d((11, 11), (self.config["apply_blur_inside"], self.config["apply_blur_inside"]))
                    # result_rgba = gauss(result_rgba.float())
                    result_rgba = torch.cat((gauss(result),gauss(gauss(ren_mask[:,None].float()))),1)
                else:
                    result_rgba = torch.cat((result,ren_mask[:,None]),1)
                renders.append(result_rgba)

                if ti < 1:
                    rotation_matrix = torch.matmul(rotation_matrix, rotation_matrix_step)

            renders = torch.stack(renders,1).contiguous()
            all_renders.append(renders)
        all_renders = torch.stack(all_renders,1).contiguous()
        return all_renders

def generate_rotation(rotation_current, my_rot, steps=3):
    step = angle_axis_to_rotation_matrix(torch.Tensor([my_rot])).to(rotation_current.device)
    step_back = angle_axis_to_rotation_matrix(torch.Tensor([-np.array(my_rot)])).to(rotation_current.device)
    for ki in range(steps):
        rotation_current = torch.matmul(rotation_current, step_back)
    rotation_matrix_join = torch.cat((step[None],rotation_current[None]),1)[None]
    return rotation_matrix_join

def generate_all_views(best_model, static_translation, rotation_matrix, rendering, small_step, extreme_step=None, num_small_steps=1):
    rendering.config["fmo_steps"] = 2
    if not extreme_step is None:
        ext_renders = rendering(static_translation, generate_rotation(rotation_matrix,extreme_step,0), best_model["vertices"], best_model["face_features"], best_model["texture_maps"])
        ext_renders_neg = rendering(static_translation, generate_rotation(rotation_matrix,-np.array(extreme_step),0), best_model["vertices"], best_model["face_features"], best_model["texture_maps"])
    rendering.config["fmo_steps"] = num_small_steps+1
    renders = rendering(static_translation, generate_rotation(rotation_matrix,small_step,0), best_model["vertices"], best_model["face_features"], best_model["texture_maps"])
    renders_neg = rendering(static_translation, generate_rotation(rotation_matrix,-np.array(small_step),0), best_model["vertices"], best_model["face_features"], best_model["texture_maps"])
    if not extreme_step is None:
        all_renders = torch.cat((ext_renders_neg[:,:,-1:], torch.flip(renders_neg[:,:,1:],[2]), renders, ext_renders[:,:,-1:]),2)
    else:
        all_renders = torch.cat((torch.flip(renders_neg[:,:,1:],[2]), renders),2)
    return all_renders.detach().cpu().numpy()[0,0].transpose(2,3,1,0)
  

def generate_novel_views(best_model, config):
    width = best_model["renders"].shape[-1]
    height = best_model["renders"].shape[-2]
    config["erode_renderer_mask"] = 7
    config["fmo_steps"] = best_model["renders"].shape[-4]
    rendering = RenderingKaolin(config, best_model["faces"], width, height).to(best_model["translation"].device)
    static_translation = best_model["translation"].clone()
    static_translation[:,:,:,1] = static_translation[:,:,:,1] + 0.5*static_translation[:,:,:,0]
    static_translation[:,:,:,0] = 0

    quaternion = best_model["quaternion"][:,:1].clone()
    rotation_matrix = angle_axis_to_rotation_matrix(quaternion_to_angle_axis(quaternion[:,0,1]))    
    rotation_matrix_step = angle_axis_to_rotation_matrix(quaternion_to_angle_axis(quaternion[:,0,0])/config["fmo_steps"]/2)
    for ki in range(int(config["fmo_steps"]/2)): rotation_matrix = torch.matmul(rotation_matrix, rotation_matrix_step)
    
    vertical =   generate_all_views(best_model, static_translation, rotation_matrix, rendering, [math.pi/2/9,0,0], [math.pi/3,0,0], 3)
    horizontal = generate_all_views(best_model, static_translation, rotation_matrix, rendering, [0,math.pi/2/9,math.pi/2/9], [0,math.pi/3,math.pi/3], 3)
    joint = generate_all_views(best_model, static_translation, rotation_matrix, rendering, [math.pi/2/9,math.pi/2/9,math.pi/2/9], [math.pi/3,math.pi/3,math.pi/3], 3)

    # steps = 1
    # config["fmo_steps"] = 2*steps+1
    # rot_joint = generate_rotation(rotation_matrix,[math.pi/3,0,0],steps)
    # hor_ext_renders = rendering(static_translation, rot_joint, best_model["vertices"], best_model["face_features"], best_model["texture_maps"])
    # rot_joint = generate_rotation(rotation_matrix,[0,math.pi/3,math.pi/3],steps)
    # ver_ext_renders = rendering(static_translation, rot_joint, best_model["vertices"], best_model["face_features"], best_model["texture_maps"])
    
    # steps = 3
    # config["fmo_steps"] = 2*steps+1
    # rot_joint = generate_rotation(rotation_matrix,[math.pi/2/9,0,0],steps)
    # hor_renders = rendering(static_translation, rot_joint, best_model["vertices"], best_model["face_features"], best_model["texture_maps"])
    # rot_joint = generate_rotation(rotation_matrix,[0,math.pi/2/9,math.pi/2/9],steps)
    # ver_renders = rendering(static_translation, rot_joint, best_model["vertices"], best_model["face_features"], best_model["texture_maps"])
    
    # vertical = torch.cat((ver_ext_renders[:,:,:1],ver_renders,ver_ext_renders[:,:,-1:]),2)
    # horizontal = torch.cat((hor_ext_renders[:,:,:1],hor_renders,hor_ext_renders[:,:,-1:]),2)

    # renders = rendering(best_model["translation"], best_model["quaternion"], best_model["vertices"], best_model["face_features"], best_model["texture_maps"])
    # save_image(renders[0,0], 'orig.png')
    # save_image(horizontal[0,0], 'hor.png')
    # save_image(vertical[0,0], 'ver.png')
    # save_image(depth[0,0], 'depth.png')

    return horizontal, vertical, joint


def generate_video_views(best_model, config):
    width = best_model["renders"].shape[-1]
    height = best_model["renders"].shape[-2]
    config["erode_renderer_mask"] = 7
    config["fmo_steps"] = best_model["renders"].shape[-4]
    rendering = RenderingKaolin(config, best_model["faces"], width, height).to(best_model["translation"].device)
    static_translation = best_model["translation"].clone()
    static_translation[:,:,:,1] = static_translation[:,:,:,1] + 0.5*static_translation[:,:,:,0]
    static_translation[:,:,:,0] = 0

    quaternion = best_model["quaternion"][:,:1].clone()
    rotation_matrix = angle_axis_to_rotation_matrix(quaternion_to_angle_axis(quaternion[:,0,1]))    
    rotation_matrix_step = angle_axis_to_rotation_matrix(quaternion_to_angle_axis(quaternion[:,0,0])/config["fmo_steps"]/2)
    for ki in range(int(config["fmo_steps"]/2)): rotation_matrix = torch.matmul(rotation_matrix, rotation_matrix_step)
    
    views = generate_all_views(best_model, static_translation, rotation_matrix, rendering, [math.pi/2/9/10,0,0], None, 45)
    return views

def generate_tsr_video(best_model, config, steps = 8):
    width = best_model["renders"].shape[-1]
    height = best_model["renders"].shape[-2]
    config["erode_renderer_mask"] = 7
    config["fmo_steps"] = steps
    rendering = RenderingKaolin(config, best_model["faces"], width, height).to(best_model["translation"].device)
    renders = rendering(best_model["translation"], best_model["quaternion"], best_model["vertices"], best_model["face_features"], best_model["texture_maps"])
    tsr = renders.detach().cpu().numpy()[0,0].transpose(2,3,1,0)
    return tsr