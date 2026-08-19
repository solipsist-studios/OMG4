import gsplat
import torch

def render(viewpoint_camera, pc, w2c, K_s):
    means = pc.means
    t=viewpoint_camera.timestamp
    dt = t - pc.times
    means_t = means + dt * pc.velocities
    # degree-2 motion: center(t) = x + v*dt + a*dt^2 (zero-init accels
    # leave degree-1 checkpoints bit-identical)
    if getattr(pc, 'accels', None) is not None and pc.accels.shape[0] == means.shape[0]:
        means_t = means_t + (dt * dt) * pc.accels
    scales = torch.exp(pc.scales)
    _temporal_opacity =  torch.exp(-0.5 * ((t - pc.times) / pc.durations.exp()) ** 2)
    quats = pc.quats
    sh_degree = pc.sh_degree

    xyzt = pc.contract_to_unisphere(means_t.clone().detach(), torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device='cuda'))
    cont_feature = pc.mlp_cont(xyzt)

    space_feature = torch.cat([cont_feature, pc._features_static],dim=-1)
    view_feature = torch.cat([cont_feature, pc._features_view],dim=-1)

    shs = pc.mlp_view(view_feature).reshape(-1,47,3).float()
    dc = pc.mlp_dc(space_feature).reshape(-1,1,3).float()
    opacities = torch.sigmoid(pc.mlp_opacity(space_feature).float())                
    colors = torch.cat([dc, shs], dim=1)

    w2c = torch.tensor(w2c[0]).float().to("cuda")
    K_s = torch.tensor(K_s).float().to("cuda")


    image, alpha, meta  = gsplat.rasterization(
        means=means_t,
        quats=quats,
        scales=scales,
        opacities=(opacities * _temporal_opacity).squeeze(),  
        colors=colors,
        viewmats=w2c.unsqueeze(0),
        Ks=K_s.unsqueeze(0),
        sh_degree=sh_degree,
        width=int(viewpoint_camera.width),
        height=int(viewpoint_camera.height))
    
    image = image.clone().clamp(0, 1)
    
    return image, alpha, meta