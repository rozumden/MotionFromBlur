import h5py
import os, glob
import numpy as np
import torch
from torchvision import transforms

import sys
sys.path.insert(0, '../MultiFrameShapeFromBlur/fmobenchmark')
from benchmark.reporters import GroundTruthProcessor
from benchmark.loaders_helpers import *

from main_settings import *
from utils import *
from helpers.torch_helpers import *
import scipy.io
from PIL import Image, ImageDraw, ImageSequence
import cv2
import time

from DeFMO.models.encoder import EncoderCNN
from DeFMO.models.rendering import RenderingCNN
import trimesh
import zipfile
import json
from dataloaders.ziploader import *

class SynDataset(torch.utils.data.Dataset):
	def __init__(self, config, device):
		self.config = config
		self.files = os.listdir(os.path.join(g_syn_folder))

	def __len__(self):
		return len(self.files)

	def __getitem__(self, index):
		zip_name = self.files[index]
		with zipfile.ZipFile(os.path.join(g_syn_folder, zip_name), 'r') as zip_handle:
			params = json.loads(zip_handle.comment)
			info = json.loads(zip_handle.getinfo("0000.webp").comment)
			img = Image.open(zip_handle.open("0000.webp"))
			model3d = trimesh.exchange.obj.load_obj(zip_handle.open("model.obj"),skip_materials=True)
			if 'geometry' in model3d:
				vertices = np.empty((0,3))
				faces = np.empty((0,3))
				for key in model3d["geometry"].keys():
					faces = np.r_[faces, model3d["geometry"][key]["faces"] + vertices.shape[0]] 
					vertices = np.r_[vertices, model3d["geometry"][key]["vertices"]]
			else:
				vertices = model3d["vertices"]
				faces = model3d["faces"]
			mesh = trimesh.Trimesh(vertices,faces)
			info["mesh"] = mesh
		frames = ImageSequence.all_frames(img)
		n_blurs = len(params["blurs"])
		n_frames = params["n_frames"]
		blurs, frames = frames[:n_blurs], frames[n_blurs:(n_blurs+n_frames)]

		hs_frames = torch.stack([transforms.ToTensor()(fr) for fr in frames],0).contiguous()[None]

		bgr_array = []
		bg_zip_handle = ZipLoader(g_bg_folder,balance_subdirs=True)
		bgs = bg_zip_handle.get_fixed_seq(max(n_blurs,3))
		
		for i, (blur, bg) in enumerate(zip(blurs, bgs[-n_blurs:])):
			bg = Image.open(bg_zip_handle.zip.open(bg))
			bgr_array.append(bg.resize(blur.size).convert(blur.mode))
			blurs[i] = Image.alpha_composite(bgr_array[-1], blur)

		inputs = torch.stack([transforms.ToTensor()(fr)[:3] for fr in blurs],0).contiguous()
		bgr_array = torch.stack([transforms.ToTensor()(fr)[:3] for fr in bgr_array],0).contiguous()
		inputs = torch.cat((inputs, bgr_array), 1)
	
		return inputs, hs_frames, info


class RealDataset(torch.utils.data.Dataset):
	def __init__(self, config, device):
		mode = config["dataset"]
		input_frames = config["input_frames"]
		if mode == 'tbd':
			ind = 30
			seq = 11
		elif mode == 'tbd3d':
			ind = 30
			seq = 5
		elif mode == 'tbdfalling':
			ind = 2
			seq = 2
		self.input, self.hs_frames = get_sample(range(ind,ind+input_frames),seq,config,device)	

	def __len__(self):
		return 1

	def __getitem__(self, index):
		return self.input, self.hs_frames

def get_sample(inds, seq, config, device):
	nmed = 7
	if config["dataset"] == 'tbd':
		files = get_tbd_dataset(g_tbd_folder)
	elif config["dataset"] == 'tbd3d':
		files = get_tbd3d_dataset(g_tbd3d_folder)
	elif config["dataset"] == 'tbdfalling':
		files = get_falling_dataset(g_falling_folder)

	if config["input_defmo"]:
		g_saved_models_folder = '../defmo/saved_models/'
		encoder = EncoderCNN().to(device)
		rendering = RenderingCNN().to(device)
		encoder.load_state_dict(torch.load(os.path.join(g_saved_models_folder, 'encoder_best.pt')))
		rendering.load_state_dict(torch.load(os.path.join(g_saved_models_folder, 'rendering_best.pt')))
		encoder.train(False)
		rendering.train(False)
		times = torch.linspace(0.01,0.99,8).to(device)

	seqpath = files[seq]
	gtp = GroundTruthProcessor(seqpath,seq,nmed)

	gt_batch_crop_all = []
	hs_frames = torch.Tensor([])
	input_batch_all = torch.Tensor([])
	for framenum in inds:
		gt_traj, radius, bbox = gtp.get_trajgt(framenum)
		gt_hs = gtp.get_hs(framenum)
		I, B = gtp.get_img(framenum)
		
		bbox = extend_bbox_uniform(bbox,radius,I.shape)
		bbox_tight = bbox_fmo(extend_bbox_uniform(bbox.copy(),10,I.shape),gt_hs,B)
		
		if config["input_defmo"]:
			bbox = extend_bbox(bbox_tight.copy(),4.0*np.max(radius),g_resolution_y/g_resolution_x,I.shape)
			im_crop = crop_resize(I, bbox, (g_resolution_x, g_resolution_y))
			bgr_crop = crop_resize(B, bbox, (g_resolution_x, g_resolution_y))
			input_batch = torch.cat((transforms.ToTensor()(im_crop), transforms.ToTensor()(bgr_crop)), 0).unsqueeze(0).float()
			with torch.no_grad():
				latent = encoder(input_batch.to(device))
				renders = rendering(latent,times[None].repeat(latent.shape[0],1))

			renders_rgba = renders[0].data.cpu().detach().numpy().transpose(2,3,1,0)
			est_hs = rev_crop_resize(renders_rgba,bbox,np.zeros((I.shape[0],I.shape[1],4)))

		bbox = extend_bbox(bbox_tight.copy(),1.0*np.max(radius),g_resolution_y/g_resolution_x,I.shape)
		im_crop = crop_resize(I, bbox, (g_resolution_x, g_resolution_y))
		bgr_crop = crop_resize(B, bbox, (g_resolution_x, g_resolution_y))
		# bbox = extend_bbox_uniform(bbox_tight.copy(),1.0*np.max(radius),I.shape)
		# im_crop = crop_only(I, bbox)
		# bgr_crop = crop_only(B, bbox)
		input_batch = torch.cat((transforms.ToTensor()(im_crop), transforms.ToTensor()(bgr_crop)), 0).unsqueeze(0).float()

		input_batch_all = torch.cat( (input_batch_all, input_batch), 0)
		if config["input_defmo"]:
			# defmo_masks = crop_only(est_hs, bbox)
			defmo_masks = crop_resize(est_hs, bbox, (g_resolution_x, g_resolution_y))
			defmo_masks_crop = torch.zeros((1,gt_hs.shape[-1],4,input_batch.shape[-2],input_batch.shape[-1]))
			for tti in range(gt_hs.shape[-1]):
				defmo_masks_crop[0,tti] = transforms.ToTensor()(defmo_masks[:,:,:,tti])
			hs_frames = torch.cat( (hs_frames, defmo_masks_crop), 0)
		else:
			gt_hs = gt_hs[bbox[0]:bbox[2], bbox[1]:bbox[3], :, :]
			gt_batch_crop = np.ones((g_resolution_y,g_resolution_x,4,gt_hs.shape[-1]))
			for tti in range(gt_hs.shape[-1]):
				gt_batch_crop[:,:,:3,tti] = cv2.resize(gt_hs[...,tti] , (g_resolution_x, g_resolution_y), interpolation = cv2.INTER_CUBIC)
			gt_batch_crop_all.append(gt_batch_crop)
	
	
	if not config["input_defmo"]:
		hs_frames = torch.stack([torch.Tensor(fr) for fr in gt_batch_crop_all],0).contiguous().permute(0,4,3,1,2)
		gt_hs_a = (torch.sum(torch.abs(hs_frames[:,:,:3] - input_batch_all[:,None,3:]),2) / 0.1 )[:,:,None]
		gt_hs_a[gt_hs_a > 1] = 1
		hs_frames[:,:,3:] = gt_hs_a
	
	return input_batch_all, hs_frames


def get_sample_benchmark(inds, seq, config, device):
	nmed = 7
	if config["dataset"] == 'tbd':
		files = get_tbd_dataset(g_tbd_folder)
	elif config["dataset"] == 'tbd3d':
		files = get_tbd3d_dataset(g_tbd3d_folder)
		files.sort()
	elif config["dataset"] == 'tbdfalling':
		files = get_falling_dataset(g_falling_folder)
		files.sort()
	else:
		files = np.array(glob.glob(os.path.join(dataset_folder, config["dataset"], 'imgs/*')))
	gtp = GroundTruthProcessor(files[seq],seq,nmed)
	
	Is = []
	Bs = []
	bboxs = []
	gts = []
	for ind in inds:
		I, B = gtp.get_img(ind)

		if gtp.w_trajgt:
			gt_traj, radius, bbox = gtp.get_trajgt(ind)
			gt_hs = gtp.get_hs(ind)
			bbox = extend_bbox_uniform(bbox,radius,I.shape)
			bbox_tight = bbox_fmo(extend_bbox_uniform(bbox.copy(),10,I.shape),gt_hs,B)
		else:
			bbox_tight, radius = fmo_detect_maxarea(I,B)
			gtp.nsplits = 8
			gt_hs = []

		Is.append(I)
		Bs.append(B)
		bboxs.append(bbox_tight)
		gts.append(gt_hs)

	return Is, Bs, bboxs, gtp.nsplits, radius, gts
