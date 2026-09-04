import os
import torch
from torchvision.utils import save_image
from torch.utils.data import Dataset
from torchvision import datasets
from utils.general_utils import PILtoTorch
from PIL import Image
import numpy as np

def _composite_and_resize(image_path, resolution, bg):
    """Decode one RGBA frame, composite over `bg`, resize to `resolution`,
    and return the uint8 HxWx3 array. This is the exact pixel chain the
    per-sample loader used, so cached and uncached paths agree bit for bit."""
    with Image.open(image_path) as image_load:
        im_data = np.array(image_load.convert("RGBA"))
    # float32, not float64: the original chain allocated several
    # float64 copies per sample (198 MB each at 2656x2324), which made
    # the loader the bottleneck at production resolution — GPU idle at
    # 0% while the workers saturated the CPU. Results differ only by
    # +/-1/255 from float rounding, far below capture noise.
    alpha = im_data[:, :, 3:4].astype(np.float32) * np.float32(1.0 / 255.0)
    rgb = im_data[:, :, :3].astype(np.float32)
    bg255 = np.asarray(bg, dtype=np.float32) * np.float32(255.0)
    if bg255.any():
        arr = rgb * alpha + bg255 * (np.float32(1.0) - alpha)
    else:
        arr = rgb * alpha        # black background: the bg term is a no-op
    image_load = Image.fromarray(arr.astype(np.uint8), "RGB")
    return np.array(image_load.resize(resolution))


def _mem_available_bytes():
    """MemAvailable from /proc/meminfo (counts reclaimable page cache, which
    SC_AVPHYS_PAGES does not); None when unavailable."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _decode_job(job):
    return _composite_and_resize(*job)


def build_image_cache(viewpoint_stack, white_background, workers=None):
    """Decode every meta_only view once into a shared-memory uint8 tensor of
    shape (N, 3, H, W). Built in the parent before DataLoader workers fork,
    so the workers inherit one copy instead of each re-decoding PNGs for
    the whole run (~60k decodes at 30k iterations x batch 2).

    Returns None (and prints why) when the views differ in size or the
    cache would exceed half of the currently available RAM."""
    metas = [c for c in viewpoint_stack if c.meta_only]
    if not metas:
        return None
    sizes = {tuple(c.resolution) for c in metas}
    if len(sizes) != 1:
        print(f"[image cache] skipped: {len(sizes)} distinct resolutions")
        return None
    (w, h), = sizes
    n = len(viewpoint_stack)
    nbytes = n * 3 * h * w
    avail = _mem_available_bytes()
    if avail is not None and nbytes > avail * 0.5:
        print(f"[image cache] skipped: {nbytes / 1e9:.1f} GB needed, "
              f"{avail / 1e9:.1f} GB available")
        return None
    bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
    cache = torch.empty((n, 3, h, w), dtype=torch.uint8).share_memory_()
    jobs = [(c.image_path, tuple(c.resolution), bg) for c in viewpoint_stack]
    workers = workers or max(1, min(16, (os.cpu_count() or 2) - 2))
    print(f"[image cache] decoding {n} views at {w}x{h} ({nbytes / 1e9:.1f} GB) "
          f"with {workers} workers")
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(workers) as pool:
        for i, arr in enumerate(pool.map(_decode_job, jobs, chunksize=4)):
            cache[i] = torch.from_numpy(arr).permute(2, 0, 1)
    return cache


class CameraDataset(Dataset):
    
    def __init__(self, viewpoint_stack, white_background, image_cache=None):
        self.viewpoint_stack = viewpoint_stack
        self.bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
        # Optional (N, 3, H, W) uint8 tensor from build_image_cache, indexed
        # in step with viewpoint_stack.
        self.image_cache = image_cache
        
    def __getitem__(self, index):
        viewpoint_cam = self.viewpoint_stack[index]
        if viewpoint_cam.meta_only:
            if self.image_cache is not None:
                viewpoint_image = self.image_cache[index].float().div_(255.0)
            else:
                arr = _composite_and_resize(viewpoint_cam.image_path, viewpoint_cam.resolution, self.bg)
                viewpoint_image = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)
            viewpoint_image = viewpoint_image.clamp_(0.0, 1.0)
        else:
            viewpoint_image = viewpoint_cam.image
            
        return viewpoint_image, viewpoint_cam
    
    def __len__(self):
        return len(self.viewpoint_stack)
