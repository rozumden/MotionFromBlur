import numpy as np
from skimage.draw import line_aa
from skimage.measure import label, regionprops
import cv2
import yaml

def write_video(array4d, path, fps=6):
	array4d[array4d < 0] = 0
	array4d[array4d > 1] = 1
	out = cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*"MJPG"), fps, (array4d.shape[1], array4d.shape[0]),True)
	for ki in range(array4d.shape[3]):
		out.write( (array4d[:,:,[2,1,0],ki] * 255).astype(np.uint8) )
	out.release()

def load_config(config_name):
    with open(config_name) as file:
        config = yaml.safe_load(file)
    return config

def imread(name):
	img = cv2.imread(name,cv2.IMREAD_UNCHANGED)
	if img.shape[2] == 3:
		return img[:,:,[2,1,0]]/255
	else:
		return img[:,:,[2,1,0,3]]/65535

def imwrite(im, name = './tmp.png'):
	im[im<0]=0
	im[im>1]=1
	cv2.imwrite(name, im[:,:,[2,1,0]]*255)

def fmo_detect_maxarea(I,B,th=0.1):
	## simulate FMO detector -> find approximate location of FMO
	dI = (np.sum(np.abs(I-B),2) > th).astype(float)
	labeled = label(dI)
	regions = regionprops(labeled)
	ind = -1
	maxarea = 0
	for ki in range(len(regions)):
		if regions[ki].area > maxarea:
			ind = ki
			maxarea = regions[ki].area
	if ind == -1:
		return [], 0
	bbox = np.array(regions[ind].bbox).astype(int)
	return bbox, regions[ind].minor_axis_length

def write_trajectory(Img, traj, vals = [0,0,1]):
	for kk in range(traj.shape[1]-1):
		Img = renderTraj(np.c_[traj[:,kk], traj[:,kk+1]-traj[:,kk]][::-1], Img, vals)
	return Img

def renderTraj(pars, H, vals = [0,0,1]):
	## Input: pars is either 2x2 (line) or 2x3 (parabola)
	if pars.shape[1] == 2:
		pars = np.concatenate( (pars, np.zeros((2,1))),1)
		ns = 2
	else:
		ns = 5

	ns = np.max([2, ns])

	rangeint = np.linspace(0,1,ns)
	for timeinst in range(rangeint.shape[0]-1):
		ti0 = rangeint[timeinst]
		ti1 = rangeint[timeinst+1]
		start = pars[:,0] + pars[:,1]*ti0 + pars[:,2]*(ti0*ti0)
		end = pars[:,0] + pars[:,1]*ti1 + pars[:,2]*(ti1*ti1)
		start = np.round(start).astype(np.int32)
		end = np.round(end).astype(np.int32)
		rr, cc, val = line_aa(start[0], start[1], end[0], end[1])
		valid = np.logical_and(np.logical_and(rr < H.shape[0], cc < H.shape[1]), np.logical_and(rr > 0, cc > 0))
		rr = rr[valid]
		cc = cc[valid]
		val = val[valid]
		if len(H.shape) > 2:
			H[rr, cc, 0] = vals[0]*val
			H[rr, cc, 1] = vals[1]*val
			H[rr, cc, 2] = vals[2]*val
		else:
			H[rr, cc] = val 
	return H

def extend_bbox(bbox,ext,aspect_ratio,shp):
	height, width = bbox[2] - bbox[0], bbox[3] - bbox[1]
			
	h2 = height + ext

	h2 = int(np.ceil(np.ceil(h2 / aspect_ratio) * aspect_ratio))
	w2 = int(h2 / aspect_ratio)

	wdiff = w2 - width
	wdiff2 = int(np.round(wdiff/2))
	hdiff = h2 - height
	hdiff2 = int(np.round(hdiff/2))

	bbox[0] -= hdiff2
	bbox[2] += hdiff-hdiff2
	bbox[1] -= wdiff2
	bbox[3] += wdiff-wdiff2
	bbox[bbox < 0] = 0
	bbox[2] = np.min([bbox[2], shp[0]-1])
	bbox[3] = np.min([bbox[3], shp[1]-1])
	return bbox

def extend_bbox_uniform(bbox,ext,shp):
	bbox[0] -= ext
	bbox[2] += ext
	bbox[1] -= ext
	bbox[3] += ext
	bbox[bbox < 0] = 0
	bbox[2] = np.min([bbox[2], shp[0]-1])
	bbox[3] = np.min([bbox[3], shp[1]-1])
	return bbox

def rgba2hs(rgba, bgr):
	return rgba[:,:,:3]*rgba[:,:,3:] + bgr[:,:,:,None]*(1-rgba[:,:,3:])

def crop_resize(Is, bbox, res):
	if Is is None:
		return None
	rev_axis = False
	if len(Is.shape) == 3:
		rev_axis = True
		Is = Is[:,:,:,np.newaxis]
	imr = np.zeros((res[1], res[0], Is.shape[2], Is.shape[3]))
	for kk in range(Is.shape[3]):
		im = Is[bbox[0]:bbox[2], bbox[1]:bbox[3], :, kk]
		imr[:,:,:,kk] = cv2.resize(im, res, interpolation = cv2.INTER_CUBIC)
	if rev_axis:
		imr = imr[:,:,:,0]
	return imr

def rev_crop_resize(inp, bbox, I):
	est_hs = np.tile(I.copy()[:,:,:,np.newaxis],(1,1,1,inp.shape[3]))
	for hsk in range(inp.shape[3]):
		est_hs[bbox[0]:bbox[2], bbox[1]:bbox[3],:,hsk] = cv2.resize(inp[:,:,:,hsk], (bbox[3]-bbox[1],bbox[2]-bbox[0]), interpolation = cv2.INTER_CUBIC)
	return est_hs

def crop_only(Is, bbox):
	if Is is None:
		return None
	return Is[bbox[0]:bbox[2], bbox[1]:bbox[3]]