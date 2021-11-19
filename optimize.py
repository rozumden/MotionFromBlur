import argparse
import os
import torch
import time
from utils import *
from dataloaders.tbd_loader import *
from msfb import *
from models.rendering import generate_novel_views

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False, default="configs/config_optimize.yaml")
    parser.add_argument("--sfb", required=False, default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    write_imgs = False
    write_meshes = True
    write_short_videos = True
    write_long_videos = True
    do_sfb = args.sfb
    if do_sfb:
        config["iterations"] = int(config["iterations"]/2)

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
    
    # for ki in range(len(bboxs_tight)):  # REMEMBER TO DELETE
        # bboxs_tight[ki] = extend_bbox_nonuniform(bboxs_tight[ki],[150, 150],Is[0].shape[:2])
    if g_do_real:
        bboxs_tight[-1] = extend_bbox_nonuniform(bboxs_tight[-1],[11, 0],Is[0].shape[:2])
        for ki in range(len(bboxs_tight)):
            bboxs_tight[ki][2] = 200

    bbox_use = bboxs_tight[0].copy()
    for bbox_tight_one in bboxs_tight:
        # bbox_tight_one = extend_bbox_nonuniform(bbox_tight_one.copy(),[150, 150],Is[0].shape[:2]) # REMEMBER TO DELETE
        bbox_use[:2] = np.c_[bbox_use[:2], bbox_tight_one[:2]].min(1)
        bbox_use[2:] = np.c_[bbox_use[2:], bbox_tight_one[2:]].max(1)
    if g_do_real:
        bbox_use = np.array([0,0,Is[0].shape[0],Is[0].shape[1]])

    traj_final = crop_only(Bs[0].copy(),bbox_use)
    inc_factor = 1
    if g_do_real:
        traj_final = crop_only(Is[-1].copy(),bbox_use)
        traj_final[:100] = crop_only(Is[2].copy(),bbox_use)[:100]
        inc_factor = 2
    traj_final = resize_only(traj_final, inc_factor)
    defmo_long = []
    est_long = []
    hs_long = []
    ss_long = []
    for fi_use in range(len(Is)):
        sfb = MultiFrameShapeFromBlur(config, device)

        if do_sfb:
            fb_name = 'sfb'
            fi_steps = 1
            I = Is[fi_use:(fi_use+fi_steps)]
            B = Bs[fi_use:(fi_use+fi_steps)]
            gt_hs = gts_hs[fi_use:(fi_use+fi_steps)]
            bbox_tight = bboxs_tight[fi_use:(fi_use+fi_steps)]
            config["frame_inds"] = config_orig["frame_inds"][fi_use:(fi_use+fi_steps)]
        else:
            fb_name = 'mfb'
            I = Is
            B = Bs
            gt_hs = gts_hs
            bbox_tight = bboxs_tight

        t0 = time.time()
        if apply_direct:
            best_model = sfb.apply_sfb(input_batch, hs_frames)
        else:
            best_model = sfb.apply(I,B,bbox_tight,nsplits,radius,None)
        print('{:4d} epochs took {:.2f} seconds, best model loss {:.4f}'.format(config["iterations"], (time.time() - t0)/1, best_model["value"]))

        write_obj_mesh(best_model["vertices"][0].cpu().numpy(), best_model["faces"], best_model["face_features"][0].cpu().numpy(), os.path.join(tmp_folder,'paper_imgs','imgs','mesh.obj'))
        save_image(best_model["texture_maps"], os.path.join(tmp_folder,'paper_imgs','imgs','tex.png'))
        do_write_mesh = lambda fi0, ti0, best_model, config: write_obj_mesh(mesh_rotate(best_model,config,fi0,ti0)[0].cpu().numpy(), best_model["faces"], best_model["face_features"][0].cpu().numpy(), os.path.join(tmp_folder,'paper_imgs','imgs','mesh_{:02d}_{:02d}.obj'.format(config["frame_inds"][fi0],ti0)))
        points_exp = get_3dtraj(best_model,config)
        if write_meshes:
            write_obj_traj_exp(points_exp, os.path.join(tmp_folder,'paper_imgs','imgs','traj_exp.obj'))
            write_obj_traj_exp(points_exp[1:-1], os.path.join(tmp_folder,'paper_imgs','imgs','traj_noexp.obj'))
        # points = []
        # trans_start = best_model["translation"][:,:,0,0][0,0]
        # points.append(trans_start.cpu().numpy())
        # for transi in range(best_model["translation"].shape[2]):
        #     trans_start = trans_start + best_model["translation"][:,:,transi,1:][0,0].sum(0)
        #     points.append(trans_start.cpu().numpy())
        # write_obj_traj(points, os.path.join(tmp_folder,'paper_imgs','traj_full.obj'))
        # breakpoint()
        traj_prev = []
        for fi in range(len(I)):
            if write_meshes:
                do_write_mesh(fi,0,best_model, config)
                do_write_mesh(fi,7,best_model, config)

            est = rev_crop_resize(best_model["renders"][0,fi].transpose(2,3,1,0), sfb.bbox, np.zeros((I[fi].shape[0],I[fi].shape[1],4)))
            
            # for ki in range(est.shape[3]): est[:,:,3,ki] = gaussian_filter(est[:,:,3,ki], sigma=1.0)

            # bbox_use = extend_bbox_nonuniform(bbox_tight[fi].copy(),[5, 15],I[fi].shape[:2])
            # bbox_use = bbox_tight[fi]

            est_hs = rgba2hs(est, B[fi])
            defmo = rev_crop_resize(best_model["hs_frames"][0,fi].transpose(2,3,1,0), sfb.bbox, np.zeros((I[fi].shape[0],I[fi].shape[1],4)))
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

            # traj = renders2traj(torch.from_numpy(best_model["renders"][:,fi]), 'cpu')[0].T
            est_tight_resized = resize_only(est_tight, inc_factor)
            traj = renders2traj(torch.from_numpy(est_tight_resized.transpose(3,2,0,1)[None].copy()), 'cpu')[0].T
            # mid_id = int(est_tight_resized.shape[3]/2)
            # if fi == 1:
            #     mid_id = 0
            # elif fi == len(I)-1:
            #     mid_id = est_tight_resized.shape[3]-1
            # else:
            #     mid_id = []
            # if mid_id != []:
            #     traj_final = rgba2hs(est_tight_resized[:,:,:,mid_id:(mid_id+1)].copy(), traj_final)[:,:,:,0]
            if traj_prev != []:
                traj_noexp = np.c_[traj_prev[:,-1], traj[:,0]]
                traj_final = write_trajectory(traj_final, traj_noexp[[1,0]], [1,1,0])
            traj_final = write_trajectory(traj_final, traj[[1,0]], [0,0,1])
            traj_prev = traj

            if write_imgs:
                for ki in range(nsplits):
                    imwrite(est_hs_tight[:,:,:,ki],os.path.join(tmp_folder,'paper_imgs','imgs','{}_{:02d}_{:02d}_{}{}.png'.format(config["dataset"],config["seq_ind"],fb_name,config["frame_inds"][fi],ki)))
                    imwrite(defmo_hs_tight[:,:,:,ki],os.path.join(tmp_folder,'paper_imgs','imgs','{}_{:02d}_{:02d}_defmo{}.png'.format(config["dataset"],config["seq_ind"],config["frame_inds"][fi],ki)))
                    imwrite(1-est_tight[:,:,[3,3,3],ki],os.path.join(tmp_folder,'paper_imgs','imgs','{}_{:02d}_{:02d}_m{}.png'.format(config["dataset"],config["seq_ind"],config["frame_inds"][fi],ki)))
                    imwrite(F_white[:,:,:,ki],os.path.join(tmp_folder,'paper_imgs','imgs','{}_{:02d}_{:02d}_f{}.png'.format(config["dataset"],config["seq_ind"],config["frame_inds"][fi],ki)))
                    if gt_available:
                        imwrite(gt_hs_tight[:,:,:,ki],os.path.join(tmp_folder,'paper_imgs','imgs','{}_{:02d}_{:02d}_hs{}.png'.format(config["dataset"],config["seq_ind"],config["frame_inds"][fi],ki)))
            if write_short_videos:
                write_video(est_hs_tight, os.path.join(tmp_folder,'paper_imgs','imgs','{}_{:02d}_{:02d}_{}.avi'.format(config["dataset"],config["seq_ind"],config["frame_inds"][fi],fb_name)))
                write_video(defmo_hs_tight, os.path.join(tmp_folder,'paper_imgs','imgs','{}_{:02d}_{:02d}_defmo.avi'.format(config["dataset"],config["seq_ind"],config["frame_inds"][fi])))
                write_video(1-est_tight[:,:,[3,3,3]], os.path.join(tmp_folder,'paper_imgs','imgs','{}_{:02d}_{:02d}_m.avi'.format(config["dataset"],config["seq_ind"],config["frame_inds"][fi])))
                write_video(F_white, os.path.join(tmp_folder,'paper_imgs','imgs','{}_{:02d}_{:02d}_f.avi'.format(config["dataset"],config["seq_ind"],config["frame_inds"][fi])))
                if gt_available:
                    write_video(gt_hs_tight, os.path.join(tmp_folder,'paper_imgs','imgs','{}_{:02d}_{:02d}_hs.avi'.format(config["dataset"],config["seq_ind"],config["frame_inds"][fi])))

            imwrite(crop_only(B[fi],bbox_use),os.path.join(tmp_folder,'paper_imgs','imgs','{}_{:02d}_{:02d}_bgr.png'.format(config["dataset"],config["seq_ind"],config["frame_inds"][fi])))
            imwrite(crop_only(I[fi],bbox_use),os.path.join(tmp_folder,'paper_imgs','imgs','{}_{:02d}_{:02d}_im.png'.format(config["dataset"],config["seq_ind"],config["frame_inds"][fi])))
            ss_long.append(crop_only(I[fi],bbox_use)[...,None])

            Irecon = crop_only(B[fi],bbox_use) * (1 - est_tight[:,:,3:].mean(3)) + (est_tight[:,:,:3]*est_tight[:,:,3:]).mean(3)
            imwrite(Irecon,os.path.join(tmp_folder,'paper_imgs','imgs','{}_{:02d}_{:02d}_est_im.png'.format(config["dataset"],config["seq_ind"],config["frame_inds"][fi]))) 
            imwrite((1 - est_tight[:,:,3:].mean(3)) + (est_tight[:,:,:3]*est_tight[:,:,3:]).mean(3),os.path.join(tmp_folder,'paper_imgs','imgs','{}_{:02d}_{:02d}_est_fs.png'.format(config["dataset"],config["seq_ind"],config["frame_inds"][fi])))
            imwrite((1 - est_tight[:,:,[3,3,3]].mean(3)),os.path.join(tmp_folder,'paper_imgs','imgs','{}_{:02d}_{:02d}_est_ms.png'.format(config["dataset"],config["seq_ind"],config["frame_inds"][fi])))
           
            Irecon_defmo = crop_only(B[fi],bbox_use) * (1 - defmo_tight[:,:,3:].mean(3)) + (defmo_tight[:,:,:3]*defmo_tight[:,:,3:]).mean(3)
            imwrite(Irecon_defmo,os.path.join(tmp_folder,'paper_imgs','imgs','{}_{:02d}_{:02d}_defmo_im.png'.format(config["dataset"],config["seq_ind"],config["frame_inds"][fi])))
            imwrite((1 - defmo_tight[:,:,3:].mean(3)) + (defmo_tight[:,:,:3]*defmo_tight[:,:,3:]).mean(3),os.path.join(tmp_folder,'paper_imgs','imgs','{}_{:02d}_{:02d}_defmo_fs.png'.format(config["dataset"],config["seq_ind"],config["frame_inds"][fi])))

            if gt_available:
                print('SfB: PSNR {:.3f}, SSIM {:.3f}'.format(calculate_psnr(gt_hs_tight, est_hs_tight), calculate_ssim(gt_hs_tight, est_hs_tight)))
                print('DeFMO: PSNR {:.3f}, SSIM {:.3f}'.format(calculate_psnr(gt_hs_tight, defmo_hs_tight), calculate_ssim(gt_hs_tight, defmo_hs_tight)))
                print('IM: PSNR {:.3f}, SSIM {:.3f}'.format(calculate_psnr(gt_hs_tight, crop_only(I[fi],bbox_use)), calculate_ssim(gt_hs_tight, crop_only(I[fi],bbox_use))))
            
            print('DeFMO IoU(0,-1) = {:.3f}'.format(calciou_masks(defmo[:,:,3,0], defmo[:,:,3,-1])))
            print('SfB IoU(0,-1) = {:.3f}'.format(calciou_masks(est[:,:,3,0], est[:,:,3,-1])))

            if False:
                config["apply_blur_inside"] = 0.7
                if write_imgs:
                    hor, ver, joint = generate_novel_views(best_model, config)
                    hor_orig = crop_only(rgba2hs(rev_crop_resize(hor, sfb.bbox, np.zeros((I[fi].shape[0],I[fi].shape[1],4))),np.ones(B[fi].shape)), bbox_use)
                    ver_orig = crop_only(rgba2hs(rev_crop_resize(ver, sfb.bbox, np.zeros((I[fi].shape[0],I[fi].shape[1],4))),np.ones(B[fi].shape)), bbox_use)
                    joint_orig = crop_only(rgba2hs(rev_crop_resize(joint, sfb.bbox, np.zeros((I[fi].shape[0],I[fi].shape[1],4))),np.ones(B[fi].shape)), bbox_use)
                    for ki in range(hor.shape[3]):
                        imwrite(hor_orig[:,:,:,ki],os.path.join(tmp_folder,'paper_imgs','imgs','{}_{:02d}_{:02d}_hor{}.png'.format(config["dataset"],config["seq_ind"],config["frame_inds"][fi],ki)))
                        imwrite(ver_orig[:,:,:,ki],os.path.join(tmp_folder,'paper_imgs','imgs','{}_{:02d}_{:02d}_ver{}.png'.format(config["dataset"],config["seq_ind"],config["frame_inds"][fi],ki)))
                        imwrite(joint_orig[:,:,:,ki],os.path.join(tmp_folder,'paper_imgs','imgs','{}_{:02d}_{:02d}_j{}.png'.format(config["dataset"],config["seq_ind"],config["frame_inds"][fi],ki)))
                else:
                    views = generate_video_views(best_model, config)
                    views_orig = crop_only(rgba2hs(rev_crop_resize(views, sfb.bbox, np.zeros((I[fi].shape[0],I[fi].shape[1],4))),np.ones(B[fi].shape)), bbox_use)
                    write_video(views_orig, os.path.join(tmp_folder,'paper_imgs','imgs','{}_{:02d}_{:02d}_novel.avi'.format(config["dataset"],config["seq_ind"],config["frame_inds"][fi])),fps=20)
                    # write_video(hor_orig, os.path.join(tmp_folder,'paper_imgs','imgs','{}_{:02d}_{:02d}_hor.avi'.format(config["dataset"],config["seq_ind"],config["frame_inds"][fi])))
                    # write_video(ver_orig, os.path.join(tmp_folder,'paper_imgs','imgs','{}_{:02d}_{:02d}_ver.avi'.format(config["dataset"],config["seq_ind"],config["frame_inds"][fi])))
                    # write_video(joint_orig, os.path.join(tmp_folder,'paper_imgs','imgs','{}_{:02d}_{:02d}_j.avi'.format(config["dataset"],config["seq_ind"],config["frame_inds"][fi])))
                    frc = 5
                    tsr = generate_tsr_video(best_model, config, est.shape[3]*frc)
                    tsr_orig = crop_only(rgba2hs(rev_crop_resize(tsr, sfb.bbox, np.zeros((I[fi].shape[0],I[fi].shape[1],4))),B[fi]), bbox_use)
                    if gt_available and est_flip:
                        tsr_orig = tsr_orig[:,:,:,::-1]
                    write_video(tsr_orig, os.path.join(tmp_folder,'paper_imgs','imgs','{}_{:02d}_{:02d}_tsr.avi'.format(config["dataset"],config["seq_ind"],config["frame_inds"][fi])),fps=6*frc)
                    write_video(tsr_orig, os.path.join(tmp_folder,'paper_imgs','imgs','{}_{:02d}_{:02d}_tsrslow.avi'.format(config["dataset"],config["seq_ind"],config["frame_inds"][fi])))
        
        if not do_sfb:
            break  

    imwrite(traj_final,os.path.join(tmp_folder,'paper_imgs','imgs','{}_{:02d}_traj.png'.format(config["dataset"],config["seq_ind"])))
    if write_long_videos:
        fps_base = 1
        fps_factor = 8
        write_video(np.concatenate(ss_long,-1), os.path.join(tmp_folder,'paper_imgs','imgs','{}_{:02d}_ss.avi'.format(config["dataset"],config["seq_ind"])),fps=fps_base)
        write_video(np.concatenate(est_long,-1), os.path.join(tmp_folder,'paper_imgs','imgs','{}_{:02d}_{}.avi'.format(config["dataset"],config["seq_ind"],fb_name)),fps=fps_base*fps_factor)
        write_video(np.concatenate(defmo_long,-1), os.path.join(tmp_folder,'paper_imgs','imgs','{}_{:02d}_defmo.avi'.format(config["dataset"],config["seq_ind"])),fps=fps_base*fps_factor)
        if gt_available:
            write_video(np.concatenate(hs_long,-1), os.path.join(tmp_folder,'paper_imgs','imgs','{}_{:02d}_hs.avi'.format(config["dataset"],config["seq_ind"])),fps=fps_base*fps_factor)
    breakpoint()



if __name__ == "__main__":
    main()