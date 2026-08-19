"""Smoke test: decode a comp.xz with the OMG4 repo code and render one frame
through the real CUDA rasterizer (validates the omg4 conda env end-to-end).

Usage:
    python script/omg4_env_smoke.py <model_dir>/comp.xz <model_dir>/cameras.json \
        [out.png] [--cam-prefix cam00] [--timestamp 2.5] [--width W] [--height H]

Render size defaults to the width/height recorded in cameras.json for the
selected camera; pass --width/--height to override.
"""
import argparse, json, math, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from gaussian_renderer import render
from utils.compress_utils import load_comp

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('comp_xz', help='encoded model (comp.xz)')
parser.add_argument('cameras_json', help='cameras.json from the same model dir')
parser.add_argument('out', nargs='?', default='omg4_smoke.png', help='output image path')
parser.add_argument('--cam-prefix', default='cam00', help='img_name prefix selecting the camera to render')
parser.add_argument('--timestamp', type=float, default=2.5, help='scene time to render')
parser.add_argument('--width', type=int, default=None, help='render width (default: from cameras.json)')
parser.add_argument('--height', type=int, default=None, help='render height (default: from cameras.json)')
args = parser.parse_args()

comp = load_comp(args.comp_xz)
pc = GaussianModel(sh_degree=3, gaussian_dim=4, time_duration=[0.0, 10.0], rot_4d=True, force_sh_3d=False, sh_degree_t=2)
pc.construct_net(train=False)   # pre-build MLPs; decode(train=True) path needs training tensors
pc.decode(comp)
pc.active_sh_degree = 3
pc.active_sh_degree_t = 2
print('decoded:', pc.get_xyz.shape[0], 'gaussians')

cams = json.load(open(args.cameras_json))
c = [c for c in cams if c['img_name'].startswith(args.cam_prefix)][0]
W2C_R = np.array(c['rotation'])  # cameras.json stores C2W rotation
pos = np.array(c['position'])
R = W2C_R  # Camera class stores R as C2W (transposed W2C), matching camera_to_JSON
T = -W2C_R.T @ pos
W = args.width or c['width']
H = args.height or c['height']
fovy = 2*math.atan(c['height']/(2*abs(c['fy'])))
fovx = 2*math.atan(c['width']/(2*abs(c['fx'])))
img = torch.zeros(3, H, W)
cam = Camera(colmap_id=0, R=R, T=T, FoVx=fovx, FoVy=fovy, image=img, gt_alpha_mask=None,
             image_name='smoke', uid=0, timestamp=args.timestamp, resolution=(W, H))

class Pipe: pass
pipe = Pipe()
pipe.convert_SHs_python = False
pipe.compute_cov3D_python = False
pipe.debug = False
pipe.env_map_res = 0

bg = torch.zeros(3, device='cuda')
out = render(cam, pc, pipe, bg)
im = out['render'].clamp(0, 1).detach().cpu().numpy().transpose(1, 2, 0)
print('rendered:', im.shape, 'mean', im.mean().round(4), 'max', im.max().round(3))
from PIL import Image
Image.fromarray((im*255).astype(np.uint8)).save(args.out)
print('OK')
