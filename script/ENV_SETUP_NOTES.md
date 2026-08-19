# OMG4 on a modern toolchain (example: RTX 5090 / sm_120, CUDA 13.2)

Working conda env: `omg4` (created by script/setup_omg4_env.sh; set
`TORCH_CUDA_ARCH_LIST` for your GPU — the values below are from an sm_120 box)

- python 3.11, torch 2.13.0+cu130, cupy-cuda13x, cuml-cu13 (26.06), dahuffman,
  plyfile, lpips, omegaconf, imageio, kornia, torchmetrics, imagesize
- Built from this repo with TORCH_CUDA_ARCH_LIST="12.0":
  diff-gaussian-rasterization, simple-knn, pointops2
- tiny-cuda-nn NOT installed: the repo's TorchFallbackMLP is used instead,
  which matches the published comp.xz checkpoints (they were trained with the
  fallback: nn.Linear with biases, LeakyReLU 0.1).

Gotchas found and fixed:
1. `diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py` is
   missing from the OMG4 repo (and upstream fudan-zvg/4d-gaussian-splatting),
   so `pip install ./diff-gaussian-rasterization` fails. A reconstructed
   wrapper matching this repo's ext.cpp bindings (14-tuple forward with
   accumulation outputs) was added locally.
2. `GaussianModel.decode()` calls `construct_net(train=True)` if MLPs are
   missing, which crashes on a fresh model (expects training tensors). Call
   `pc.construct_net(train=False)` before `decode()` — see
   script/omg4_env_smoke.py.

Smoke test (decode comp.xz + render one frame via the CUDA rasterizer):

    conda run -n omg4 python script/omg4_env_smoke.py \
        <model_dir>/comp.xz <model_dir>/cameras.json /tmp/omg4_smoke.png

Training runs as per README (`python train.py ...`) inside `conda activate omg4`
once a dataset (data/N3V/...) and the pretrained 4D-GS seed weights are in place.

## FoV-sentinel rendering bug (found 2026-07-09)

The dataset loaders build Cameras with `FoVx = FoVy = -1` (sentinel) and real
intrinsics in `cx/cy/fl_x/fl_y`. `gaussian_renderer.render()` still computes
`tanfovx = tan(FoVx * 0.5) = tan(-0.5)`, so the rasterizer's EWA Jacobian uses
focal `W/(2*tan(0.5)) ~= 1237px` while the pixel projection uses `fl ~= 730px`
(N3V at half res). Every splat footprint is rendered ~1.69x wider and ~1.27x
taller than geometry says — during training too, so trained scales compensate
downward. Result: checkpoints look correct only in this repo's renderer and
thin/streaky in any geometrically-correct renderer (gsplat, web viewers).

- Reproducing the repo's own renders requires the sentinel camera
  (verified bit-exact, 58.8 dB).
- For export to .omg4: `xz_to_omg4.py --scale_boost 1.4672` (isotropic
  sqrt(1.6942*1.2707)) approximates the trained appearance.
- For NEW trainings on custom data, fix the sentinel before training (set
  FoVx/FoVy from fl in scene/cameras.py or dataset_readers) so the model is
  geometry-consistent and needs no boost. The FTGS variant (gsplat-based)
  does not have this bug.
