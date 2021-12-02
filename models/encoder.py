import torch
import torch.nn as nn
from kornia.geometry.conversions import angle_axis_to_rotation_matrix, quaternion_to_angle_axis, rotation_matrix_to_quaternion
from kornia.geometry.conversions import quaternion_to_rotation_matrix, rotation_matrix_to_angle_axis

def mesh_normalize(vertices):
    mesh_max = torch.max(vertices, dim=1, keepdim=True)[0]
    mesh_min = torch.min(vertices, dim=1, keepdim=True)[0]
    mesh_middle = (mesh_min + mesh_max) / 2
    vertices = vertices - mesh_middle
    bs = vertices.shape[0]
    mesh_biggest = torch.max(vertices.view(bs, -1), dim=1)[0]
    vertices = vertices / mesh_biggest.view(bs, 1, 1) 
    return vertices
    
class EncoderMulti(nn.Module):
    def __init__(self, config, ivertices, faces, face_features, width, height):
        super(EncoderMulti, self).__init__()
        self.config = config
        self.translation = nn.Parameter(torch.zeros(1,config["number_of_pieces"],1,config["number_of_terms"],3))
        self.quaternion = nn.Parameter(torch.ones(1,config["number_of_pieces"],1,2,4))
        if self.config["predict_vertices"]:
            self.vertices = nn.Parameter(torch.zeros(1,ivertices.shape[0],3))
        self.register_buffer('face_features', torch.from_numpy(face_features).unsqueeze(0).type(self.translation.dtype))
        self.texture_map = nn.Parameter(torch.ones(1,3,self.config["texture_size"],self.config["texture_size"]))
        ivertices = torch.from_numpy(ivertices).unsqueeze(0).type(self.translation.dtype)
        ivertices = mesh_normalize(ivertices)
        self.register_buffer('ivertices', ivertices)
        self.aspect_ratio = height/width
        self.exposure_fraction = nn.Parameter(5*torch.ones(1,1))

    def forward(self):
        output = {}
        output["exp"] = nn.Sigmoid()(self.exposure_fraction)
        translation_all = []
        quaternion_all = []
        for frmi in range(self.translation.shape[1]):
            translation = self.translation[:,frmi,None]
            quaternion = self.quaternion[:,frmi]
            translation_all.append(translation)
            quaternion_all.append(quaternion)

        output["translation"] = torch.stack(translation_all,2).contiguous()[:,:,:,0]
        output["quaternion"] = torch.stack(quaternion_all,1).contiguous()[:,:,0]
        if self.config["predict_vertices"]:
            vertices = self.ivertices + self.vertices
            output["vertices"] = vertices - vertices.mean(1)[:,None,:]
        else:
            output["vertices"] = self.ivertices

        output["face_features"] = self.face_features
        output["texture_maps"] = self.texture_map
        return output



