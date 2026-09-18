 # Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import dnnlib
import PIL
#import clip
from camera_utils import LookAtPoseSampler
import kornia
import logging

def rot2eul(R):
    beta = -np.arcsin(R[2,0])
    alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
    gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
    return np.array((alpha, beta, gamma))


def prepare_w_plus(initial_w, w_avg, num_ws, device):
    """Convert a W or W+ pivot into a validated [1, num_ws, C] tensor."""
    start_w = w_avg if initial_w is None else initial_w
    if torch.is_tensor(start_w):
        start_w = start_w.detach().cpu().numpy()
    start_w = np.asarray(start_w, dtype=np.float32)
    if start_w.ndim == 2:
        start_w = start_w[:, None, :]
    if start_w.ndim != 3 or start_w.shape[0] != 1:
        raise ValueError(f'Expected W/W+ shape [1, 1|{num_ws}, C], got {start_w.shape}')
    if start_w.shape[1] == 1:
        start_w = np.repeat(start_w, num_ws, axis=1)
    elif start_w.shape[1] != num_ws:
        raise ValueError(f'Expected 1 or {num_ws} style vectors, got {start_w.shape[1]}')
    return torch.as_tensor(start_w, dtype=torch.float32, device=device)


def _w_gradient_norm(loss, w_opt):
    """Measure d(loss)/d(W+) without modifying accumulated gradients."""
    if not torch.is_tensor(loss) or not loss.requires_grad:
        return 0.0
    gradient = torch.autograd.grad(
        loss, w_opt, retain_graph=True, allow_unused=True)[0]
    if gradient is None:
        return 0.0
    return float(gradient.detach().float().norm().cpu())


def calculate_ssim(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Return mean RGB SSIM for equally shaped BCHW images in [0, 1]."""
    if first.shape != second.shape:
        raise ValueError(
            f'SSIM inputs must have equal shapes, got {first.shape} and {second.shape}')
    if first.ndim != 4 or first.shape[1] != 3:
        raise ValueError(f'Expected RGB BCHW tensors for SSIM, got {first.shape}')
    return kornia.metrics.ssim(
        first.clamp(0.0, 1.0), second.clamp(0.0, 1.0),
        window_size=11, max_val=1.0).mean()


def save_stage_result(
        G, c, target, w_pivot, latent_reference, deid_loss, attr_loss,
        outdir, stage_name, optimization_seconds, device):
    """Render and score the actual state returned by an optimization stage."""
    stage_dir = os.path.join(outdir, stage_name)
    os.makedirs(stage_dir, exist_ok=True)
    started = time.perf_counter()
    was_training = G.training
    G.eval()
    with torch.no_grad():
        generated_raw = G.synthesis(
            w_pivot.to(device), c.to(device), noise_mode='const')['image']
        generated = (generated_raw + 1) * (255 / 2)
        target_full = target.unsqueeze(0).to(device=device, dtype=torch.float32)

        vis = (generated_raw.permute(0, 2, 3, 1) * 127.5 + 128)
        vis = vis.clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(vis[0].cpu().numpy(), 'RGB').save(
            os.path.join(stage_dir, 'final.png'))

        pixel_mse = F.mse_loss(generated / 255.0, target_full / 255.0)
        ssim = calculate_ssim(generated / 255.0, target_full / 255.0)
        generated_256 = F.interpolate(generated, (256, 256), mode='area')
        target_256 = F.interpolate(target_full, (256, 256), mode='area')
        with dnnlib.util.open_url('./networks/vgg16.pt') as f:
            vgg16 = torch.jit.load(f).eval().to(device)
        generated_features = vgg16(
            generated_256.clone(), resize_images=False, return_lpips=True)
        target_features = vgg16(
            target_256.clone(), resize_images=False, return_lpips=True)
        vgg_distance = (generated_features - target_features).square().sum()

        generated_norm = generated_256 * 2 / 255.0 - 1
        target_norm = target_256 * 2 / 255.0 - 1
        arcface_similarity = deid_loss.similarity(
            generated_norm, target_norm).mean()
        target_gender, target_expression = attr_loss.predict_classes(target_norm)
        generated_gender, generated_expression = attr_loss.predict_classes(
            generated_norm)
        latent_distance = F.mse_loss(
            w_pivot.to(device), latent_reference.to(device))

    if was_training:
        G.train()
    evaluation_seconds = time.perf_counter() - started
    metrics = {
        'stage': stage_name,
        'pixel_mse': float(pixel_mse.cpu()),
        # Mean RGB SSIM at the native 512x512 evaluation resolution.
        'ssim': float(ssim.cpu()),
        # This is the repository's StyleGAN VGG perceptual distance, not the
        # learned LPIPS package metric; the explicit name prevents ambiguity.
        'vgg_distance': float(vgg_distance.cpu()),
        'arcface_similarity': float(arcface_similarity.cpu()),
        'gender_agreement': bool((generated_gender == target_gender).all().cpu()),
        'expression_agreement': bool(
            (generated_expression == target_expression).all().cpu()),
        'target_gender_class': int(target_gender[0].cpu()),
        'generated_gender_class': int(generated_gender[0].cpu()),
        'target_expression_class': int(target_expression[0].cpu()),
        'generated_expression_class': int(generated_expression[0].cpu()),
        'latent_mse_to_stage_reference': float(latent_distance.cpu()),
        'optimization_seconds': float(optimization_seconds),
        'evaluation_seconds': float(evaluation_seconds),
        'total_stage_seconds': float(optimization_seconds + evaluation_seconds),
    }
    with open(os.path.join(stage_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    return metrics


def project(
        G,
        c,
        outdir,
        target: torch.Tensor,  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
        *,
        num_steps=1000,
        w_avg_samples=10000,
        initial_learning_rate=0.01,
        initial_noise_factor=0.05,
        lr_rampdown_length=0.25,
        lr_rampup_length=0.05,
        noise_ramp_length=0.75,
        regularize_noise_weight=1e3,
        verbose=False,
        device: torch.device,
        initial_w=None,
        image_log_step=1,
        w_name: str,
        deid_loss,
        attr_loss,
        lamda_origin,
        lambda_deid,
        lambda_gender,
        lambda_expr,
        lambda_latent,
        stage_name="pre",
        enable_privacy=True,
        optimize_noise=True,
        lambda_perceptual=0.0,
        lambda_identity_preservation=0.0,
        full_resolution_pixel=False,
        inject_latent_noise=True,
        gradient_log_interval=0,
        save_progress_images=True,
):
    outdir = os.path.join(outdir, stage_name)
    os.makedirs(outdir, exist_ok=True)
    log_out_dir = os.path.join(outdir, "log.txt")
    
    logging.basicConfig(
        filename=log_out_dir, filemode='w', level=logging.INFO, force=True)
    #################################
    #the hyperparameters: weight of i_loss, weight of d_loss, and weight of original_loss
    
    weight_of_original_loss = lamda_origin
    weight_of_deid_loss = lambda_deid
    weight_of_gender_loss = lambda_gender
    weight_of_expr_loss = lambda_expr
    weight_of_latent_reg = lambda_latent
    
    logging.info("weight_of_original_loss: "+str(weight_of_original_loss))
    logging.info("weight_of_deid_loss: "+str(weight_of_deid_loss))
    logging.info("weight_of_gender_loss: "+str(weight_of_gender_loss))
    logging.info("weight_of_expr_loss: "+str(weight_of_expr_loss))
    logging.info("weight_of_latent_reg: "+str(weight_of_latent_reg))
    logging.info("enable_privacy: "+str(enable_privacy))
    logging.info("optimize_noise: "+str(optimize_noise))
    logging.info("lambda_perceptual: "+str(lambda_perceptual))
    logging.info("lambda_identity_preservation: "+str(lambda_identity_preservation))
    logging.info("full_resolution_pixel: "+str(full_resolution_pixel))
    logging.info("inject_latent_noise: "+str(inject_latent_noise))
    logging.info("gradient_log_interval: "+str(gradient_log_interval))
    gradient_log_path = os.path.join(outdir, 'gradient_norms.jsonl')

    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)
    G = copy.deepcopy(G).eval().requires_grad_(False).to(device).float() # type: ignore

    # Used both when computing W statistics and for randomized side views.
    # Keep it outside the cache branch so cached W statistics follow the same
    # rendering path as freshly computed statistics.
    camera_lookat_point = torch.tensor(
        G.rendering_kwargs['avg_camera_pivot'], device=device)

    w_avg_path = './w_avg.npy'
    w_std_path = './w_std.npy'
    if os.path.exists(w_avg_path) and os.path.exists(w_std_path):
        print(f'Loading cached W midpoint and stddev from {w_avg_path} and {w_std_path}...')
        w_avg = np.load(w_avg_path).astype(np.float32)
        w_std = float(np.load(w_std_path))
    else:
        print(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
        z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
        cam2world_pose = LookAtPoseSampler.sample(3.14 / 2, 3.14 / 2, camera_lookat_point,
                                                  radius=G.rendering_kwargs['avg_camera_radius'], device=device)
        focal_length = 4.2647  # FFHQ's FOV
        intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
        c_samples = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        c_samples = c_samples.repeat(w_avg_samples, 1)

        w_samples = G.mapping(torch.from_numpy(z_samples).to(device), c_samples)  # [N, L, C]
        w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
        w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
        w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5
        np.save(w_avg_path, w_avg)
        np.save(w_std_path, np.array(w_std, dtype=np.float32))

    # Setup noise inputs.
    noise_bufs = {name: buf for (name, buf) in G.backbone.synthesis.named_buffers() if 'noise_const' in name}

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    target_images_orginal_illu = target_images
    
    if target_images_orginal_illu.shape[2] > 256:
        target_images = F.interpolate(target_images_orginal_illu, size=(256, 256), mode='area')
    vgg16 = None
    target_features = None
    if lambda_perceptual != 0:
        # The TorchScript VGG detector normalizes its argument in place. Always
        # pass a clone so the [0,255] tensor remains valid for other losses.
        with dnnlib.util.open_url('./networks/vgg16.pt') as f:
            vgg16 = torch.jit.load(f).eval().to(device)
        target_features = vgg16(
            target_images.clone(), resize_images=False, return_lpips=True)

    latent_reference = prepare_w_plus(
        initial_w, w_avg, G.backbone.mapping.num_ws, device)
    w_opt = latent_reference.clone().detach().requires_grad_(True)

    optimized_parameters = [w_opt]
    if optimize_noise:
        optimized_parameters += list(noise_bufs.values())
    optimizer = torch.optim.Adam(optimized_parameters, betas=(0.9, 0.999),
                                 lr=0.1)
    # Init noise.
    if optimize_noise:
        for buf in noise_bufs.values():
            buf[:] = torch.randn_like(buf)
            buf.requires_grad = True

    for step in tqdm(range(num_steps)):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        w_noise = (torch.randn_like(w_opt) * w_noise_scale
                   if inject_latent_noise else torch.zeros_like(w_opt))
        ws = (w_opt + w_noise)
        synth_images = G.synthesis(ws,c, noise_mode='const')['image']
        #synth_images size: [1, 3, 512, 512]
            
        if save_progress_images and step % image_log_step == 0:
            with torch.no_grad():
                vis_img = (synth_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

                PIL.Image.fromarray(vis_img[0].cpu().numpy(), 'RGB').save(f'{outdir}/{step}.png')

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images_original = (synth_images + 1) * (255 / 2)  # original size is 512
        if synth_images_original.shape[2] > 256:
            synth_images = F.interpolate(synth_images_original, size=(256, 256), mode='area')
        synth_images_relight = F.interpolate(synth_images_original, size=(512, 512), mode='area')
     
        # Pixel-space reconstruction error.
        pixel_synth = synth_images_original if full_resolution_pixel else synth_images
        pixel_target = target_images_orginal_illu if full_resolution_pixel else target_images
        original_loss = F.mse_loss(
            pixel_synth / 255.0, pixel_target / 255.0)

        perceptual_loss = synth_images.new_zeros(())
        if lambda_perceptual != 0:
            synth_features = vgg16(
                synth_images.clone(), resize_images=False, return_lpips=True)
            perceptual_loss = (target_features - synth_features).square().sum()

        identity_preservation_loss = synth_images.new_zeros(())
        if lambda_identity_preservation != 0:
            synth_images_norm = synth_images * 2 / 255.0 - 1
            target_images_norm = target_images * 2 / 255.0 - 1
            identity_similarity = deid_loss.similarity(
                synth_images_norm, target_images_norm)
            identity_preservation_loss = (1.0 - identity_similarity).mean()

        if enable_privacy:
            # Normalise synth/target to [-1, 1] for identity & attribute losses.
            synth_images_norm = synth_images * 2 / 255.0 - 1
            target_images_norm = target_images * 2 / 255.0 - 1
            deid_val, mean_dist = deid_loss(synth_images_norm, target_images_norm)
            gender_loss, expr_loss = attr_loss(synth_images_norm, target_images_norm)
        else:
            # Stage 1 is reconstruction-only; avoid classifier forward passes.
            deid_val = synth_images.new_zeros(())
            mean_dist = synth_images.new_zeros(())
            gender_loss = synth_images.new_zeros(())
            expr_loss = synth_images.new_zeros(())
        latent_reg = F.mse_loss(w_opt, latent_reference)

        dist = (original_loss * weight_of_original_loss
                + perceptual_loss * lambda_perceptual
                + identity_preservation_loss * lambda_identity_preservation
                + deid_val * weight_of_deid_loss
                + gender_loss * weight_of_gender_loss
                + expr_loss * weight_of_expr_loss
                + latent_reg * weight_of_latent_reg)

        logging.info(str(step)+" deid_loss: "+str(deid_val.cpu().detach()))
        logging.info(str(step)+" mean_dist: "+str(mean_dist.cpu().detach()))
        logging.info(str(step)+" gender_loss: "+str(gender_loss.cpu().detach()))
        logging.info(str(step)+" expr_loss: "+str(expr_loss.cpu().detach()))
        logging.info(str(step)+" latent_reg: "+str(latent_reg.cpu().detach()))
        logging.info(str(step)+" original_loss: "+str(original_loss.cpu().detach()))
        logging.info(str(step)+" perceptual_loss: "+str(perceptual_loss.cpu().detach()))
        logging.info(str(step)+" identity_preservation_loss: "+str(identity_preservation_loss.cpu().detach()))
        ######################
        # Noise regularization.
        reg_loss = synth_images.new_zeros(())
        if optimize_noise:
            for v in noise_bufs.values():
                noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
                while True:
                    reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                    reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                    if noise.shape[2] <= 8:
                        break
                    noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight

        if (gradient_log_interval > 0 and
                (step % gradient_log_interval == 0 or step == num_steps - 1)):
            gradient_terms = {
                'pixel_or_profile_reconstruction': (
                    original_loss, weight_of_original_loss),
                'vgg_perceptual': (perceptual_loss, lambda_perceptual),
                'arcface_privacy': (deid_val, weight_of_deid_loss),
                'gender': (gender_loss, weight_of_gender_loss),
                'expression': (expr_loss, weight_of_expr_loss),
                'latent': (latent_reg, weight_of_latent_reg),
            }
            gradient_record = {'step': step, 'terms': {}}
            for term_name, (term_loss, term_weight) in gradient_terms.items():
                raw_norm = _w_gradient_norm(term_loss, w_opt)
                gradient_record['terms'][term_name] = {
                    'loss': float(term_loss.detach().cpu()),
                    'weight': float(term_weight),
                    'raw_gradient_norm': raw_norm,
                    'weighted_gradient_norm': abs(float(term_weight)) * raw_norm,
                }
            gradient_record['total_gradient_norm'] = _w_gradient_norm(loss, w_opt)
            with open(gradient_log_path, 'a', encoding='utf-8') as gradient_log:
                gradient_log.write(json.dumps(gradient_record) + '\n')

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        #torch.nn.utils.clip_grad_norm(w_opt, 1)
        optimizer.step()
        
        #logprint(f'step {step + 1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')

        # Normalize noise.
        if optimize_noise:
            with torch.no_grad():
                for buf in noise_bufs.values():
                    buf -= buf.mean()
                    buf *= buf.square().mean().rsqrt()
        torch.cuda.empty_cache()

    del G
    #torch.cuda.empty_cache()
    return w_opt


def project_pti(
        G, 
        c,
        outdir,
        target: torch.Tensor,  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
        w_pivot: torch.Tensor,
        *,
        num_steps_pti=1000,
        w_avg_samples=10000,
        initial_learning_rate=0.0003,
        initial_noise_factor=0.05,
        lr_rampdown_length=0.25,
        lr_rampup_length=0.05,
        noise_ramp_length=0.75,
        regularize_noise_weight=1e3,
        verbose=False,
        device: torch.device,
        initial_w=None,
        image_log_step=1,
        w_name: str,
        deid_loss,
        attr_loss,
        lamda_origin,
        lambda_deid,
        lambda_gender,
        lambda_expr,
        save_progress_images=True,
):
    outdir = os.path.join(outdir, "post")
    os.makedirs(outdir, exist_ok=True)
    log_out_dir = os.path.join(outdir, "log.txt")
    
    logging.basicConfig(
        filename=log_out_dir, filemode='w', level=logging.INFO, force=True)
    #################################
    #the hyperparameters: weight of i_loss, weight of d_loss, and weight of original_loss
    
    weight_of_original_loss = lamda_origin
    weight_of_deid_loss = lambda_deid
    weight_of_gender_loss = lambda_gender
    weight_of_expr_loss = lambda_expr
    
    logging.info("weight_of_original_loss: "+str(weight_of_original_loss))
    logging.info("weight_of_deid_loss: "+str(weight_of_deid_loss))
    logging.info("weight_of_gender_loss: "+str(weight_of_gender_loss))
    logging.info("weight_of_expr_loss: "+str(weight_of_expr_loss))
    ###################################
    

    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    G = copy.deepcopy(G).train().requires_grad_(True).to(device) # type: ignore
    w_pivot = w_pivot.to(device).detach()
    torch.cuda.empty_cache()
    optimizer = torch.optim.Adam(G.parameters(), betas=(0.9, 0.999),
                                 lr=initial_learning_rate)
    # Compute w stats.
    camera_lookat_point = torch.tensor(G.rendering_kwargs['avg_camera_pivot'], device=device)
    cam2world_pose = LookAtPoseSampler.sample(3.14 / 2, 3.14 / 2, camera_lookat_point,
                                                  radius=G.rendering_kwargs['avg_camera_radius'], device=device)
    focal_length = 4.2647  # FFHQ's FOV
    intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
    # Setup noise inputs.
    noise_bufs = {name: buf for (name, buf) in G.backbone.synthesis.named_buffers() if 'noise_const' in name}

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    target_images_orginal_illu = target_images
   
    if target_images_orginal_illu.shape[2] > 256:
        target_images = F.interpolate(target_images_orginal_illu, size=(256, 256), mode='area')
    deid_loss = deid_loss.to(device)
    attr_loss = attr_loss.to(device)
    target_images = target_images.to(device)
    torch.cuda.empty_cache()

    # start_w = np.repeat(start_w, G.backbone.mapping.num_ws, axis=1)
    # w_opt = torch.tensor(start_w, dtype=torch.float32, device=device,
    #                      requires_grad=True)  # pylint: disable=not-callable

    
    # Keep pretrained noise buffers fixed during PTI. They are buffers rather
    # than optimizer parameters, so randomizing them here would only introduce
    # an unrecoverable discontinuity between latent optimization and PTI.

    for step in tqdm(range(num_steps_pti)):
        # Learning rate schedule.
        t = step / num_steps_pti
        #w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # w_noise = torch.randn_like(w_opt) * w_noise_scale
        # ws = (w_opt + w_noise)
        synth_images = G.synthesis(w_pivot,c, noise_mode='const')['image'].to(device)
        #synth_images size: [1, 3, 512, 512]
            
        if save_progress_images and step % image_log_step == 0:
            with torch.no_grad():
                vis_img = (synth_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

                PIL.Image.fromarray(vis_img[0].cpu().numpy(), 'RGB').save(f'{outdir}/{step}.png')

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images_original = (synth_images + 1) * (255 / 2)  # original size is 512
        if synth_images_original.shape[2] > 256:
            synth_images = F.interpolate(synth_images_original, size=(256, 256), mode='area')
        #synth_images_relight = F.interpolate(synth_images_original, size=(512, 512), mode='area')
     
        original_loss = F.mse_loss(
            synth_images / 255.0, target_images / 255.0)

        # Normalise synth/target to [-1, 1] for identity & attribute losses
        synth_images_norm = synth_images * 2 / 255.0 - 1
        target_images_norm = target_images * 2 / 255.0 - 1

        deid_val, mean_dist = deid_loss(synth_images_norm, target_images_norm)
        gender_loss, expr_loss = attr_loss(synth_images_norm, target_images_norm)

        dist = (original_loss * weight_of_original_loss
                + deid_val * weight_of_deid_loss
                + gender_loss * weight_of_gender_loss
                + expr_loss * weight_of_expr_loss)

        logging.info(str(step)+" deid_loss: "+str(deid_val.cpu().detach()))
        logging.info(str(step)+" mean_dist: "+str(mean_dist.cpu().detach()))
        logging.info(str(step)+" gender_loss: "+str(gender_loss.cpu().detach()))
        logging.info(str(step)+" expr_loss: "+str(expr_loss.cpu().detach()))
        logging.info(str(step)+" original_loss: "+str(original_loss.cpu().detach()))
 
        torch.cuda.empty_cache()
        ######################
        loss = dist

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        #torch.nn.utils.clip_grad_norm(w_opt, 1)
        optimizer.step()
        
        #logprint(f'step {step + 1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')
        torch.cuda.empty_cache()
    return G
