#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

"""Standalone renderer for a compressed OMG4 model (comp.xz).

Loads the scene given by --config, optionally decodes an encoded
checkpoint passed via --comp_checkpoint, and writes the test cameras
(and, unless --skip_train, the train cameras) as PNG frames sorted by
timestamp. Output goes to ./output/<config-basename>/renders and
./output/<config-basename>/train_renders. Unlike test.py this computes
no metrics; it only produces frames.
"""

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.gaussian_model import GaussianModel
import lzma
import pickle
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import sys

def render_set(output_path, views, gaussians, pipeline, background):
    makedirs(output_path, exist_ok=True)

    # Access the underlying camera list and sort by timestamp to ensure chronological order
    view_list = sorted(views.viewpoint_stack, key=lambda x: x.timestamp)

    for idx, viewpoint_cam in enumerate(tqdm(view_list, desc="Rendering progress")):
        rendering = render(viewpoint_cam.cuda(), gaussians, pipeline, background)["render"]
        torchvision.utils.save_image(rendering, os.path.join(output_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset, opt, pipe, iteration, skip_train, skip_test, gaussian_dim, time_duration, num_pts, num_pts_ratio, rot_4d, force_sh_3d, comp_checkpoint, config_name):
    with torch.no_grad():
        if dataset.frame_ratio > 1:
            time_duration = [time_duration[0] / dataset.frame_ratio,  time_duration[1] / dataset.frame_ratio]

        # Disable shuffle to maintain original data order
        gaussians = GaussianModel(dataset.sh_degree, gaussian_dim=gaussian_dim, time_duration=time_duration, rot_4d=rot_4d, force_sh_3d=force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0)
        scene = Scene(dataset, gaussians, num_pts=num_pts, num_pts_ratio=num_pts_ratio, time_duration=time_duration, shuffle=False)
        gaussians.training_setup(opt)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if comp_checkpoint:
            print(f"[INFO] Loading checkpoint {comp_checkpoint}...")
            with lzma.open(comp_checkpoint, "rb") as f:
                load_dict = pickle.load(f)
            gaussians.decode(load_dict, decompress=True)
            gaussians.active_sh_degree = 3
            gaussians.active_sh_degree_t = 2
            if gaussians.env_map.device != "cuda":
                gaussians.env_map = gaussians.env_map.to("cuda")

        # Define the output directory based on the dataset name
        base_output_path = os.path.join("./output", config_name, "renders")

        if not skip_test:
             print(f"Rendering test set to {base_output_path}...")
             render_set(base_output_path, scene.getTestCameras(), gaussians, pipe, background)

        if not skip_train:
             train_output_path = os.path.join("./output", config_name, "train_renders")
             print(f"Rendering train set to {train_output_path}...")
             render_set(train_output_path, scene.getTrainCameras(), gaussians, pipe, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", type=str)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    
    parser.add_argument("--gaussian_dim", type=int, default=3)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5, 0.5])
    parser.add_argument('--num_pts', type=int, default=100_000)
    parser.add_argument('--num_pts_ratio', type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--exhaust_test", action="store_true")
    parser.add_argument("--comp_checkpoint", type=str, default = None)

    args = parser.parse_args(sys.argv[1:])

    config_name = "default_dataset"
    if args.config:
        config_name = os.path.basename(args.config).split(".")[0]
        cfg = OmegaConf.load(args.config)
        def recursive_merge(key, host):
            if isinstance(host[key], DictConfig):
                for key1 in host[key].keys():
                    recursive_merge(key1, host[key])
            else:
                assert hasattr(args, key), key
                setattr(args, key, host[key])
        for k in cfg.keys():
            recursive_merge(k, cfg)

    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(lp.extract(args), op.extract(args), pp.extract(args), args.iteration, args.skip_train, args.skip_test, 
                args.gaussian_dim, args.time_duration, args.num_pts, args.num_pts_ratio, args.rot_4d, args.force_sh_3d, args.comp_checkpoint, config_name)
