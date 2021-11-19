import os

dataset_folder = '/mnt/lascar/rozumden/dataset/'
tmp_folder = '/home.stud/rozumden/tmp/'

if not os.path.exists(dataset_folder):
	dataset_folder = '/cluster/scratch/denysr/dataset/'
	tmp_folder = '/cluster/home/denysr/tmp/'

run_folder = tmp_folder+'PyTorch/'

g_tbd_folder = dataset_folder+'TbD/'
g_tbd3d_folder = dataset_folder+'TbD-3D/'
g_falling_folder = dataset_folder+'falling_objects/'
g_wildfmo_folder = dataset_folder+'wildfmo/'
g_youtube_folder = dataset_folder+'youtube/'

g_syn_folder = dataset_folder+'synth30/'
g_bg_folder = dataset_folder+'vot2018.zip'

g_resolution_x = int(640/2)
g_resolution_y = int(480/2)

g_use_selfsupervised_timeconsistency = True
g_timeconsistency_type = 'ncc'

g_do_real = False