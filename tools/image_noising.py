import os
import json
import numpy as np
import matplotlib.pyplot as plt
import copy
from Data_Sampler_Valentinos import *
from PIL import Image
import torch
import torchvision.transforms as T
import random
from tqdm import tqdm
import argparse
import shutil
from pathlib import Path

def select_frames_to_add_noise(scene_names, annotations, noisy_scenes_perc=0.2, noisy_frames_perc=0.2, min_noisy_frms=1, max_noisy_frms=3):
    noisy_scenes = dict()
    total_scenes = len(scene_names)
    # Compute the number of noisy images
    total_noisy_imgs = round(total_scenes * noisy_scenes_perc)
    # Select the Scenes to have noise
    noisy_scenes_mask = np.zeros(total_scenes)
    noisy_scenes_mask[:total_noisy_imgs] = 1
    np.random.shuffle(noisy_scenes_mask)
    for i,sname in enumerate(scene_names):
        if int(noisy_scenes_mask[i]) == 0:
            continue
        noisy_scenes[sname] = dict()
        total_frms = len(annotations['scene_infos'][sname].keys())
        total_noisy_frms = round(total_frms*noisy_frames_perc)
        # Select the frames to have noise
        noisy_frms_mask = np.zeros(total_frms)
        noisy_frms_mask[:total_noisy_frms] = 1
        np.random.shuffle(noisy_frms_mask)
        for j, (frm_token, frm_val) in enumerate(annotations['scene_infos'][sname].items()):
            if int(noisy_frms_mask[j]) == 0:
                continue
            noisy_scenes[sname][frm_token]=set()
            # Decide for which views to have noise
            cam_views = sorted(list(frm_val['camera_sensor'].keys()))
            num_of_noisy_views = random.randint(min_noisy_frms, max_noisy_frms)
            noisy_views_idces = random.sample(range(len(cam_views)), num_of_noisy_views)
            for idx in noisy_views_idces:
                noisy_scenes[sname][frm_token].add(cam_views[idx])
    return noisy_scenes


def sample_square_patches(img_shape, patch_h, patch_w,  max_sampled_patches=1, perc_img_covered=None):
    mask = np.zeros(img_shape)
    h,w = img_shape[:2]
    # for _ in range(num_patches):
    total_pixls = h * w
    def sample_patch(mask, h=h, w=w, patch_h=patch_h, patch_w=patch_w):
        y_idx = np.random.randint(0,h-patch_h+1)
        x_idx = np.random.randint(0,w-patch_w+1)
        mask[y_idx:y_idx+patch_h, x_idx:x_idx+patch_w]=1
        return mask
    count_masked_pixels = lambda x: np.where(x[:,:,0]==1)[0].shape[0]
    if perc_img_covered is not None:
        while perc_img_covered >= count_masked_pixels(mask)/total_pixls:
            sample_patch(mask)
    else:
        for _ in range(max_sampled_patches):
            sample_patch(mask)
    return mask


def mix_images(img_0, img_1, mask):
    return np.where(mask==1, img_1, img_0)


def add_noise_to_image(img, mask, noise_type='black', noise_args={'kernel_size':(5, 9), 'sigma':(0.1, 5)}):
    img_arr = np.asarray(img)
    transf_img_arr = None
    if noise_type == 'black':
        transf_img_arr = np.zeros_like(img_arr)
    elif noise_type=='blurring':
        blurrer = T.GaussianBlur(**noise_args)
        transf_img_arr = np.asarray(blurrer(img))
    else:
        raise "Not supported Image Transformation."        
    noisy_img_arr = mix_images(img_arr, transf_img_arr, mask)
    return Image.fromarray(noisy_img_arr.astype('uint8'), 'RGB'), mask


def generate_noisy_images(scene_names, noisy_scenes, annotations, dataset_path, patch_h=20, patch_w=20,  max_sampled_patches=1, perc_img_covered=0.25, save_dir:str='.', noise_type='black', noise_args={}, save_not_noisy_imgs=True):
    for sname in tqdm(scene_names):
        not_noisy_scene = sname not in noisy_scenes
        for frm_token, frm_val in annotations['scene_infos'][sname].items():
            not_noisy_frame = not_noisy_scene or frm_token not in noisy_scenes[sname]
            for cam_view, cam_val in frm_val['camera_sensor'].items():
                not_noisy_cam_view = not_noisy_frame or cam_view not in noisy_scenes[sname][frm_token]
                if not save_not_noisy_imgs and not_noisy_cam_view:
                    continue
                # 1. Load the Image
                img = Image.open(os.path.join(dataset_path, 'samples',cam_val['img_path']))
                save_dest = os.path.join(save_dir,'samples',cam_val['img_path'])
                os.makedirs(Path(os.path.normpath(save_dest)).parent.absolute(),exist_ok=True)
                if not not_noisy_cam_view:
                    # 2. Decide where to puth the noise
                    mask = sample_square_patches(np.asarray(img).shape, patch_h=patch_h, patch_w=patch_w,  max_sampled_patches=max_sampled_patches, perc_img_covered=perc_img_covered)
                    # 3. Add the noise
                    img = add_noise_to_image(img, mask=mask,  noise_type=noise_type, noise_args=noise_args)[0]
                # 4. Save the image
                img.save(save_dest)

def copy_dataset(src_data_dir, dest_data_dir, scene_names):
    os.makedirs(dest_data_dir, exist_ok=True)
    to_copy_folders = ['maps','v1.0-trainval']
    for dir in to_copy_folders:
        src_p = os.path.join(src_data_dir, dir)
        dest_p = os.path.join(dest_data_dir, dir)
        shutil.copytree(src_p, dest_p, dirs_exist_ok=True)
    # copy the annotations.json
    src_p = os.path.join(src_data_dir, 'annotations.json')
    dest_p = os.path.join(dest_data_dir, 'annotations.json')
    shutil.copy(src_p, dest_p)

    dir_name = 'gts'
    for scene_dir in os.listdir(os.path.join(src_data_dir, dir_name)):
        if scene_dir in scene_names:
            src_p = os.path.join(src_data_dir, dir_name, scene_dir)
            dest_p = os.path.join(dest_data_dir, dir_name, scene_dir)
            shutil.copytree(src_p, dest_p, dirs_exist_ok=True)

def generate_bundle_of_noisy_imgs(annotations, train_scenes, val_scenes, bundle_noise_confs:dict, noise_selection_confs:dict, save_dir:str, dataset_path=None):
    noisy_train_scenes = select_frames_to_add_noise(train_scenes, annotations, **noise_selection_confs)
    noisy_val_scenes= select_frames_to_add_noise(val_scenes, annotations, **noise_selection_confs)
    noisy_scenes = {**noisy_train_scenes, **noisy_val_scenes}
    scene_names = set(train_scenes).union(set(val_scenes))
    for bundle_id, noise_confs in bundle_noise_confs.items():
        save_dir=os.path.join(save_dir, bundle_id, 'occ3d-nus')
        if dataset_path is not None:
            copy_dataset(dataset_path, dest_data_dir=save_dir, scene_names=scene_names)
        generate_noisy_images(scene_names, noisy_scenes, annotations, dataset_path, **noise_confs, save_dir=save_dir)

def parse_args():
    parser = argparse.ArgumentParser(
                    prog='Adder of Noise to Images',
                    description='Description',
                    epilog='Epilogue')
    parser.add_argument('--dataset_path', default='/mnt/beegfs/vpariz01/workspace/cv2/CVPR2023-3D-Occupancy-Prediction-CV2/data/occ3d-nus') 
    parser.add_argument('--save_dir', default='/mnt/beegfs/vpariz01/workspace/cv2/CVPR2023-3D-Occupancy-Prediction-CV2/noisy_images') 
    parser.add_argument('--gen_noise_case', default=list(),nargs='+',type=int)  # on/off flag
    parser.add_argument('--data_noise_case', default=1,type=int)
    return parser.parse_args()

def main():
    args = parse_args()
    # Load the annotations file
    with open(os.path.join(args.dataset_path, 'annotations.json'), 'r') as f:
        annotations = json.load(f)

    noise_selection_confs = {
        'noisy_scenes_perc':0.25,
        'noisy_frames_perc': 0.25,
        'min_noisy_frms': 1,
        'max_noisy_frms': 3
    }

    if args.data_noise_case == 2 :
        noise_selection_confs = {
            'noisy_scenes_perc':0.50,
            'noisy_frames_perc': 0.50,
            'min_noisy_frms': 1,
            'max_noisy_frms': 3
        }

    if args.data_noise_case == 3 :
        noise_selection_confs = {
            'noisy_scenes_perc':0.75,
            'noisy_frames_perc': 0.75,
            'min_noisy_frms': 1,
            'max_noisy_frms': 3
        }

    if args.data_noise_case == 4 :
        noise_selection_confs = {
            'noisy_scenes_perc':0.75,
            'noisy_frames_perc': 0.75,
            'min_noisy_frms': 3,
            'max_noisy_frms': 4
        }

    bundle_noise_confs = dict()

    def get_desc(d, nscnf, blcnf=False):
        nscnf_str = "data-perc-{}_img-perc{}_nsfrms-({},{})".format(nscnf["noisy_scenes_perc"], nscnf["noisy_frames_perc"], nscnf["min_noisy_frms"], nscnf["max_noisy_frms"])
        d_str = '{}_{}x{}_cov-{}'.format(d['noise_type'], d['patch_h'],d['patch_w'],d['perc_img_covered'])
        if blcnf is False:
            return "{}-{}".format(nscnf_str, d_str)
        else:
            return "{}-{}-{}-{}".format(nscnf_str, d_str, d['noise_args']['kernel_size'][0], d['noise_args']['sigma'][0])
    

    if 1 in args.gen_noise_case :
        noise_config= {
            'patch_h': 25, 
            'patch_w': 25,  
            'max_sampled_patches': None, 
            'perc_img_covered': 0.10, 
            'noise_type': 'black', 
            'noise_args': {}
        }
        bundle_noise_confs[get_desc(noise_config, noise_selection_confs)] = noise_config
    if 2 in args.gen_noise_case :
        noise_config= {
            'patch_h': 25, 
            'patch_w': 25,  
            'max_sampled_patches': None, 
            'perc_img_covered': 0.20, 
            'noise_type': 'black', 
            'noise_args': {}
        }
        bundle_noise_confs[get_desc(noise_config, noise_selection_confs)] = noise_config
    if 3 in args.gen_noise_case :
        noise_config= {
            'patch_h': 25, 
            'patch_w': 25,  
            'max_sampled_patches': None, 
            'perc_img_covered': 0.30, 
            'noise_type': 'black', 
            'noise_args': {}
        }
        bundle_noise_confs[get_desc(noise_config, noise_selection_confs)] = noise_config
    if 4 in args.gen_noise_case :
        noise_config= {
            'patch_h': 120, 
            'patch_w': 120,  
            'max_sampled_patches': None, 
            'perc_img_covered': 0.15, 
            'noise_type': 'blurring', 
            'noise_args': {'kernel_size':(21, 21), 'sigma':(6, 6)}
        }
        bundle_noise_confs[get_desc(noise_config, noise_selection_confs)] = noise_config
    if 5 in args.gen_noise_case :
        noise_config= {
            'patch_h': 120, 
            'patch_w': 120,  
            'max_sampled_patches': None, 
            'perc_img_covered': 0.30, 
            'noise_type': 'blurring', 
            'noise_args': {'kernel_size':(25, 25), 'sigma':(8, 8)}
        }
        bundle_noise_confs[get_desc(noise_config, noise_selection_confs)] = noise_config
    if 6 in args.gen_noise_case :
        noise_config= {
            'patch_h': 120, 
            'patch_w': 120,  
            'max_sampled_patches': None, 
            'perc_img_covered': 0.50, 
            'noise_type': 'blurring', 
            'noise_args': {'kernel_size':(25, 25), 'sigma':(8, 8)}
        }
        bundle_noise_confs[get_desc(noise_config, noise_selection_confs)] = noise_config
    if 7 in args.gen_noise_case :
        noise_config= {
            'patch_h': 120, 
            'patch_w': 120,  
            'max_sampled_patches': None, 
            'perc_img_covered': 0.75, 
            'noise_type': 'blurring', 
            'noise_args': {'kernel_size':(25, 25), 'sigma':(8, 8)}
        }
        bundle_noise_confs[get_desc(noise_config, noise_selection_confs)] = noise_config

    if 8 in args.gen_noise_case :
        noise_config= {
            'patch_h': 120, 
            'patch_w': 120,  
            'max_sampled_patches': None, 
            'perc_img_covered': 0.30, 
            'noise_type': 'blurring', 
            'noise_args': {'kernel_size':(51, 51), 'sigma':(20, 20)}
        }
        bundle_noise_confs[get_desc(noise_config, noise_selection_confs, blcnf=True)] = noise_config
    if 9 in args.gen_noise_case :
        noise_config= {
            'patch_h': 120, 
            'patch_w': 120,  
            'max_sampled_patches': None, 
            'perc_img_covered': 0.50, 
            'noise_type': 'blurring', 
            'noise_args': {'kernel_size':(51, 51), 'sigma':(20, 20)}
        }
        bundle_noise_confs[get_desc(noise_config, noise_selection_confs, blcnf=True)] = noise_config
    if 10 in args.gen_noise_case :
        noise_config= {
            'patch_h': 120, 
            'patch_w': 120,  
            'max_sampled_patches': None, 
            'perc_img_covered': 0.75, 
            'noise_type': 'blurring', 
            'noise_args': {'kernel_size':(51, 51), 'sigma':(20, 20)}
        }
        bundle_noise_confs[get_desc(noise_config, noise_selection_confs, blcnf=True)] = noise_config

    print(bundle_noise_confs)
    print(noise_selection_confs)

    train_scenes = set(annotations['train_split'])
    val_scenes = set(annotations['val_split'])
    generate_bundle_of_noisy_imgs(annotations, train_scenes, val_scenes, bundle_noise_confs=bundle_noise_confs, noise_selection_confs=noise_selection_confs, save_dir=args.save_dir, dataset_path=args.dataset_path)

if __name__ == '__main__':
    main()