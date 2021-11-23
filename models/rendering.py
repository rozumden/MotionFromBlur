import torch
import kaolin
from models.kaolin_wrapper import *
from kornia.geometry.conversions import angle_axis_to_rotation_matrix, quaternion_to_angle_axis, rotation_matrix_to_quaternion
from kornia.geometry.conversions import quaternion_to_rotation_matrix, rotation_matrix_to_angle_axis
from kornia.morphology import erosion
from utils import *  

class RenderingKaolinMulti(torch.nn.Module):
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


def get_3dtraj(best_model,config):
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

def mesh_rotate(best_model,config, frmi_needed, ti_needed):
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

def renders2traj(renders,device):
    masks = renders[:,:,-1]
    sumx = torch.sum(masks,-2)
    sumy = torch.sum(masks,-1)
    cenx = torch.sum(sumy*torch.arange(1,sumy.shape[-1]+1)[None,None].float().to(device),-1) / torch.sum(sumy,-1)
    ceny = torch.sum(sumx*torch.arange(1,sumx.shape[-1]+1)[None,None].float().to(device),-1) / torch.sum(sumx,-1)
    est_traj = torch.cat((cenx.unsqueeze(-1),ceny.unsqueeze(-1)),-1)
    return est_traj