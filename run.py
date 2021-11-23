import argparse
import os
import torch
from torchvision.utils import save_image
import time
from utils import *
from mfb import *
import cv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=False, default="./input/volleyball.avi")
    parser.add_argument("--max_frames", type=int, required=False, default=3)
    parser.add_argument("--configs", required=False, default="configs.yaml")
    parser.add_argument("--subframes", type=int, required=False, default=8)
    parser.add_argument("--output", required=False, default="./output")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.configs)
    config["fmo_steps"] = args.subframes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Start data loading
    Is = []
    Bs = []
    bboxs = []
    cap = cv2.VideoCapture(args.input)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        Is.append(frame[:,:,[2,1,0]]/255)
    B = np.median(np.asarray(Is), 0)
    Is = Is[:args.max_frames]
    config["input_frames"] = len(Is)
    radius = 0
    for ki in range(len(Is)):
        bbox, radius_temp = fmo_detect_maxarea(Is[ki],B,th=0.03)
        bboxs.append(bbox)
        Bs.append(B)
        if radius_temp > radius:
            radius = radius_temp
    bbox_use = bboxs[0].copy()
    for bbox_one in bboxs:
        bbox_one = extend_bbox(bbox_one.copy(),1.0*np.max(radius),0.75,Is[0].shape)
        bbox_use[:2] = np.c_[bbox_use[:2], bbox_one[:2]].min(1)
        bbox_use[2:] = np.c_[bbox_use[2:], bbox_one[2:]].max(1)
    bbox_use = extend_bbox_uniform(bbox_use.copy(),0.5*np.max(radius),Is[0].shape)
    # End data loading
    
    t0 = time.time()
    mfb = MotionFromBlur(config, device)
    best_model = mfb.apply(Is,Bs,bboxs,args.subframes,radius,None)
    print('{:4d} epochs took {:.2f} seconds, best model loss {:.4f}'.format(config["iterations"], (time.time() - t0)/1, best_model["value"]))

    in_long = []
    est_long = []
    traj_prev = []
    traj_final = crop_only(Bs[0].copy(),bbox_use)
    for fi in range(len(Is)):
        in_long.append(crop_only(Is[fi],bbox_use)[...,None])
        est = rev_crop_resize(best_model["renders"][0,fi].transpose(2,3,1,0), mfb.bbox, np.zeros((Is[fi].shape[0],Is[fi].shape[1],4)))
        est_hs = rgba2hs(est, Bs[fi])
        est_hs_tight = crop_only(est_hs,bbox_use)
        est_long.append(est_hs_tight)
        traj = renders2traj(torch.from_numpy(crop_only(est,bbox_use).transpose(3,2,0,1)[None].copy()), 'cpu')[0].T
        if traj_prev != []:
            traj_noexp = np.c_[traj_prev[:,-1], traj[:,0]]
            traj_final = write_trajectory(traj_final, traj_noexp[[1,0]], [1,1,0])
        traj_final = write_trajectory(traj_final, traj[[1,0]], [0,0,1])
        traj_prev = traj

    write_video(np.concatenate(in_long,-1), os.path.join(args.output,'input.avi'),fps=1)
    write_video(np.concatenate(est_long,-1), os.path.join(args.output,'mfb.avi'),fps=args.subframes)
    imwrite(traj_final,os.path.join(args.output,'traj.png'))
    imwrite(crop_only(Bs[0],bbox_use),os.path.join(args.output,'bgr.png'))
    write_obj_mesh(best_model["vertices"][0].cpu().numpy(), best_model["faces"], best_model["face_features"][0].cpu().numpy(), os.path.join(args.output,'mesh.obj'))
    save_image(best_model["texture_maps"], os.path.join(args.output,'tex.png'))
    do_write_mesh = lambda fi0, ti0, best_model, config: write_obj_mesh(mesh_rotate(best_model,config,fi0,ti0)[0].cpu().numpy(), best_model["faces"], best_model["face_features"][0].cpu().numpy(), os.path.join(args.output,'mesh_{:02d}_{:02d}.obj'.format(fi0,ti0)))
    points_exp = get_3dtraj(best_model,config)
    write_obj_traj_exp(points_exp, os.path.join(args.output,'traj_exp.obj'))
    write_obj_traj_exp(points_exp[1:-1], os.path.join(args.output,'traj_noexp.obj'))
    do_write_mesh(0,0,best_model, config)
    do_write_mesh(len(Is)-1,args.subframes-1,best_model, config)

    file = open(os.path.join(args.output,'model.mtl'),"w")
    file.write("newmtl Material.002\n")
    file.write("map_Kd tex.png\n")
    file.close()
    

if __name__ == "__main__":
    main()