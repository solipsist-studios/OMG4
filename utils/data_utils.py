import os
import torch
from torchvision.utils import save_image
from torch.utils.data import Dataset
from torchvision import datasets
from utils.general_utils import PILtoTorch
from PIL import Image
import numpy as np

class CameraDataset(Dataset):
    
    def __init__(self, viewpoint_stack, white_background):
        self.viewpoint_stack = viewpoint_stack
        self.bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
        
    def __getitem__(self, index):
        viewpoint_cam = self.viewpoint_stack[index]
        if viewpoint_cam.meta_only:
            with Image.open(viewpoint_cam.image_path) as image_load:
                im_data = np.array(image_load.convert("RGBA"))
            # float32, not float64: the original chain allocated several
            # float64 copies per sample (198 MB each at 2656x2324), which made
            # the loader the bottleneck at production resolution — GPU idle at
            # 0% while the workers saturated the CPU. Results differ only by
            # +/-1/255 from float rounding, far below capture noise.
            alpha = im_data[:, :, 3:4].astype(np.float32) * np.float32(1.0 / 255.0)
            rgb = im_data[:, :, :3].astype(np.float32)
            bg255 = np.asarray(self.bg, dtype=np.float32) * np.float32(255.0)
            if bg255.any():
                arr = rgb * alpha + bg255 * (np.float32(1.0) - alpha)
            else:
                arr = rgb * alpha        # black background: the bg term is a no-op
            image_load = Image.fromarray(arr.astype(np.uint8), "RGB")
            resized_image_rgb = PILtoTorch(image_load, viewpoint_cam.resolution)
            viewpoint_image = resized_image_rgb[:3, ...].clamp(0.0, 1.0)
            if resized_image_rgb.shape[1] == 4:
                gt_alpha_mask = resized_image_rgb[3:4, ...]
                viewpoint_image *= gt_alpha_mask
            else:
                viewpoint_image *= torch.ones((1, viewpoint_cam.image_height, viewpoint_cam.image_width))
        else:
            viewpoint_image = viewpoint_cam.image
            
        return viewpoint_image, viewpoint_cam
    
    def __len__(self):
        return len(self.viewpoint_stack)
    
