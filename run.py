import argparse
import os
import torch
import time
from utils import *
from dataloaders.tbd_loader import *
from mfb import *
from models.rendering import generate_novel_views

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=False, default="input/vol_im.png")
    parser.add_argument("--configs", required=False, default="configs.yaml")
    parser.add_argument("--subframes", required=False, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    write_imgs = False
    write_meshes = True
    write_short_videos = True
    write_long_videos = True

    config["verbose"] = True
    config["write_results"] = True
    config["input_frames"] = len(config["frame_inds"])
    config_orig = config.copy()

    print(config)
    apply_direct = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    t0 = time.time()
    print('Initialization took {:.2f} seconds'.format((time.time() - t0)/1))

    t0 = time.time()
    if apply_direct:
        training_set = RealDataset(config, device)
        input_batch, hs_frames = training_set.__getitem__(0)
    else:
        Is,Bs,bboxs_tight,nsplits,radius,gts_hs = get_sample_benchmark(config["frame_inds"], config["seq_ind"], config, device)
    print('Data loading took {:.2f} seconds'.format((time.time() - t0)/1))
    
    bbox_use = bboxs_tight[0].copy()
    for bbox_tight_one in bboxs_tight:
        bbox_use[:2] = np.c_[bbox_use[:2], bbox_tight_one[:2]].min(1)
        bbox_use[2:] = np.c_[bbox_use[2:], bbox_tight_one[2:]].max(1)

    traj_final = crop_only(Bs[0].copy(),bbox_use)
    defmo_long = []
    est_long = []
    hs_long = []
    ss_long = []
    for fi_use in range(len(Is)):
        mfb = MotionFromBlur(config, device)

        I = Is
        B = Bs
        gt_hs = gts_hs
        bbox_tight = bboxs_tight

        t0 = time.time()
        if apply_direct:
            best_model = mfb.apply_sfb(input_batch, hs_frames)
        else:
            best_model = mfb.apply(I,B,bbox_tight,nsplits,radius,None)
        print('{:4d} epochs took {:.2f} seconds, best model loss {:.4f}'.format(config["iterations"], (time.time() - t0)/1, best_model["value"]))

        write_obj_mesh(best_model["vertices"][0].cpu().numpy(), best_model["faces"], best_model["face_features"][0].cpu().numpy(), os.path.join(tmp_folder,'paper_imgs','imgs','mesh.obj'))
        save_image(best_model["texture_maps"], os.path.join(tmp_folder,'paper_imgs','imgs','tex.png'))
        do_write_mesh = lambda fi0, ti0, best_model, config: write_obj_mesh(mesh_rotate(best_model,config,fi0,ti0)[0].cpu().numpy(), best_model["faces"], best_model["face_features"][0].cpu().numpy(), os.path.join(tmp_folder,'paper_imgs','imgs','mesh_{:02d}_{:02d}.obj'.format(config["frame_inds"][fi0],ti0)))
        points_exp = get_3dtraj(best_model,config)
        if write_meshes:
            write_obj_traj_exp(points_exp, os.path.join(tmp_folder,'paper_imgs','imgs','traj_exp.obj'))
            write_obj_traj_exp(points_exp[1:-1], os.path.join(tmp_folder,'paper_imgs','imgs','traj_noexp.obj'))
       
        traj_prev = []
        for fi in range(len(I)):
            if write_meshes:
                do_write_mesh(fi,0,best_model, config)

            est = rev_crop_resize(best_model["renders"][0,fi].transpose(2,3,1,0), mfb.bbox, np.zeros((I[fi].shape[0],I[fi].shape[1],4)))
            
            est_hs = rgba2hs(est, B[fi])
            defmo = rev_crop_resize(best_model["hs_frames"][0,fi].transpose(2,3,1,0), mfb.bbox, np.zeros((I[fi].shape[0],I[fi].shape[1],4)))
            defmo_hs = rgba2hs(defmo, B[fi])
            est_tight = crop_only(est,bbox_use)
            defmo_tight = crop_only(defmo,bbox_use)
            est_hs_tight = crop_only(est_hs,bbox_use)
            defmo_hs_tight = crop_only(defmo_hs,bbox_use)
           
            gt_available = gt_hs[fi] != []
            if gt_available:
                gt_hs_tight = crop_only(gt_hs[fi],bbox_use)
                est_hs_tight, est_flip = sync_directions(est_hs_tight, gt_hs_tight)
                defmo_hs_tight, defmo_flip = sync_directions(defmo_hs_tight, gt_hs_tight)
                if est_flip:
                    est_tight = est_tight[:,:,:,::-1]
                if defmo_flip:
                    defmo_tight = defmo_tight[:,:,:,::-1]
                hs_long.append(gt_hs_tight)
            est_long.append(est_hs_tight)
            defmo_long.append(defmo_hs_tight)

            F_white = 1.0*(1 - est_tight[:,:,3:]) + (est_tight[:,:,:3]*est_tight[:,:,3:])

            est_tight_resized = resize_only(est_tight, inc_factor)
            traj = renders2traj(torch.from_numpy(est_tight_resized.transpose(3,2,0,1)[None].copy()), 'cpu')[0].T
            if traj_prev != []:
                traj_noexp = np.c_[traj_prev[:,-1], traj[:,0]]
                traj_final = write_trajectory(traj_final, traj_noexp[[1,0]], [1,1,0])
            traj_final = write_trajectory(traj_final, traj[[1,0]], [0,0,1])
            traj_prev = traj

            if write_short_videos:
                write_video(est_hs_tight, os.path.join(tmp_folder,'paper_imgs','imgs','{}_{:02d}_{:02d}_mfb.avi'.format(config["dataset"],config["seq_ind"],config["frame_inds"][fi])))
                write_video(defmo_hs_tight, os.path.join(tmp_folder,'paper_imgs','imgs','{}_{:02d}_{:02d}_defmo.avi'.format(config["dataset"],config["seq_ind"],config["frame_inds"][fi])))
                write_video(1-est_tight[:,:,[3,3,3]], os.path.join(tmp_folder,'paper_imgs','imgs','{}_{:02d}_{:02d}_m.avi'.format(config["dataset"],config["seq_ind"],config["frame_inds"][fi])))
                write_video(F_white, os.path.join(tmp_folder,'paper_imgs','imgs','{}_{:02d}_{:02d}_f.avi'.format(config["dataset"],config["seq_ind"],config["frame_inds"][fi])))
                if gt_available:
                    write_video(gt_hs_tight, os.path.join(tmp_folder,'paper_imgs','imgs','{}_{:02d}_{:02d}_hs.avi'.format(config["dataset"],config["seq_ind"],config["frame_inds"][fi])))

            imwrite(crop_only(B[fi],bbox_use),os.path.join(tmp_folder,'paper_imgs','imgs','{}_{:02d}_{:02d}_bgr.png'.format(config["dataset"],config["seq_ind"],config["frame_inds"][fi])))
            imwrite(crop_only(I[fi],bbox_use),os.path.join(tmp_folder,'paper_imgs','imgs','{}_{:02d}_{:02d}_im.png'.format(config["dataset"],config["seq_ind"],config["frame_inds"][fi])))
            ss_long.append(crop_only(I[fi],bbox_use)[...,None])


    if write_long_videos:
        fps_base = 1
        fps_factor = 8
        write_video(np.concatenate(ss_long,-1), os.path.join(tmp_folder,'paper_imgs','imgs','{}_{:02d}_ss.avi'.format(config["dataset"],config["seq_ind"])),fps=fps_base)
        write_video(np.concatenate(est_long,-1), os.path.join(tmp_folder,'paper_imgs','imgs','{}_{:02d}_mfb.avi'.format(config["dataset"],config["seq_ind"])),fps=fps_base*fps_factor)
        write_video(np.concatenate(defmo_long,-1), os.path.join(tmp_folder,'paper_imgs','imgs','{}_{:02d}_defmo.avi'.format(config["dataset"],config["seq_ind"])),fps=fps_base*fps_factor)
        if gt_available:
            write_video(np.concatenate(hs_long,-1), os.path.join(tmp_folder,'paper_imgs','imgs','{}_{:02d}_hs.avi'.format(config["dataset"],config["seq_ind"])),fps=fps_base*fps_factor)
    breakpoint()



if __name__ == "__main__":
    main()