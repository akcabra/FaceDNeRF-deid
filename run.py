
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import datetime
import json
import random
import re
import shutil
import time
from typing import Optional
import pickle
import click
import dnnlib
import numpy as np
import torch
torch.autograd.set_detect_anomaly(True)
import legacy
from torchvision.transforms import transforms
from editors import w_plus_editor
from PIL import Image
from criteria.deid_loss import DeIDLoss
from criteria.attr_loss import AttrLoss
# ----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--outdir', help='Output directory', type=str, required=True, metavar='DIR')
# @click.option('--latent_space_type', help='latent_space_type', type=click.Choice(['w', 'w_plus']), required=False, metavar='STR',
#               default='w', show_default=True)
@click.option('--image_path', help='image_path', type=str, required=True, metavar='STR', show_default=True)
@click.option('--c_path', help='camera parameters path', type=str, required=True, metavar='STR', show_default=True)
@click.option('--sample_mult', 'sampling_multiplier', type=float,
              help='Multiplier for depth sampling in volume rendering', default=2, show_default=True)
@click.option('--num_steps', 'num_steps', type=int,
              help='Number of Stage 2 de-identification steps', default=500, show_default=True)
@click.option('--num_steps_pti', 'num_steps_pti', type=int,
              help='Number of pivotal-tuning steps', default=400, show_default=True)
@click.option('--num_steps_inversion', type=int,
              help='Reconstruction-only steps before de-identification in staged mode',
              default=300, show_default=True)
@click.option('--nrr', type=int, help='Neural rendering resolution override', default=None, show_default=True)
@click.option('--lambda_origin', type=float,
              help='Pixel reconstruction loss weight', default=1.0, show_default=True)
@click.option('--pp', type=float,
              help='Privacy parameter for de-id [0=max privacy, 1=min]', default=0.0, show_default=True)
@click.option('--lambda_deid', type=float,
              help='De-identification loss weight', default=2.5, show_default=True)
@click.option('--lambda_gender', type=float,
              help='Gender preservation loss weight', default=0.01, show_default=True)
@click.option('--lambda_expr', type=float,
              help='Expression preservation loss weight', default=0.01, show_default=True)
@click.option('--lambda_latent', type=float,
              help='Latent regularizer loss weight', default=0.0016, show_default=True)
@click.option('--seed', type=int, default=42, show_default=True,
              help='Random seed for Python, NumPy, and PyTorch')
@click.option('--run_name', type=str, default=None,
              help='Optional unique run directory name (safe filename characters only)')
@click.option('--lambda_inversion_perceptual', type=float, default=0.8,
              show_default=True, help='VGG perceptual weight used only in Stage 1')
@click.option('--lambda_inversion_identity', type=float, default=0.1,
              show_default=True, help='ArcFace identity-preservation weight used only in Stage 1')
@click.option('--lambda_stage2_perceptual', type=float, default=0.4,
              show_default=True, help='VGG perceptual-preservation weight used only in Stage 2')
@click.option('--gradient_log_interval', type=click.IntRange(min=0), default=0,
              show_default=True,
              help='Log per-loss W+ gradient norms in Stage 2 every N steps; 0 disables')
@click.option('--stage2_latent_noise/--no-stage2_latent_noise', default=True,
              show_default=True,
              help='Enable decaying W+ perturbation during Stage 2')
@click.option('--stage2_pixel_resolution', type=click.Choice(['256', '512']),
              default='512', show_default=True,
              help='Resolution used by the Stage 2 pixel reconstruction loss')
@click.option('--save-progress-images/--final-images-only', default=True,
              show_default=True,
              help='Save every optimization render or only stage-final renders')
def run(
        network_pkl: str,
        outdir: str,
        sampling_multiplier: float,
        nrr: Optional[int],
        image_path:str,
        c_path:str,
        num_steps:int,
        num_steps_pti:int,
        num_steps_inversion:int,
        lambda_origin: float,
        pp: float,
        lambda_deid: float,
        lambda_gender: float,
        lambda_expr: float,
        lambda_latent: float,
        seed: int,
        run_name: Optional[str],
        lambda_inversion_perceptual: float,
        lambda_inversion_identity: float,
        lambda_stage2_perceptual: float,
        gradient_log_interval: int,
        stage2_latent_noise: bool,
        stage2_pixel_resolution: str,
        save_progress_images: bool,
):
    """Reconstruct and de-identify one EG3D-compatible face image."""
    
    
    if run_name is None:
        run_name = datetime.datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', run_name):
        raise click.BadParameter(
            'use only letters, numbers, dot, underscore, and hyphen',
            param_hint='--run_name')
    if (lambda_inversion_perceptual < 0 or lambda_inversion_identity < 0 or
            lambda_stage2_perceptual < 0):
        raise click.BadParameter('Perceptual and identity loss weights must be non-negative')

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    configuration_name = (
        f'{image_name}_deid_staged_cpp_pp{pp}_deid{lambda_deid}_'
        f'origin{lambda_origin}_gender{lambda_gender}_expr{lambda_expr}_'
        f'latent{lambda_latent}_invvgg{lambda_inversion_perceptual}_'
        f'invid{lambda_inversion_identity}_deidvgg{lambda_stage2_perceptual}_'
        f'deidpx{stage2_pixel_resolution}')
    outdir = os.path.join(outdir, configuration_name, run_name)
    try:
        os.makedirs(outdir, exist_ok=False)
    except FileExistsError as exc:
        raise click.ClickException(
            f'run directory already exists: {outdir}; choose another --run_name') from exc
    print(f'Run output: {os.path.abspath(outdir)}')

    run_config = {
        'network': network_pkl, 'image_path': image_path, 'c_path': c_path,
        'sampling_multiplier': sampling_multiplier, 'nrr': nrr,
        'num_steps': num_steps, 'num_steps_pti': num_steps_pti,
        'num_steps_inversion': num_steps_inversion, 'seed': seed,
        'pp': pp, 'lambda_deid': lambda_deid,
        'lambda_origin': lambda_origin, 'lambda_gender': lambda_gender,
        'lambda_expr': lambda_expr, 'lambda_latent': lambda_latent,
        'lambda_inversion_perceptual': lambda_inversion_perceptual,
        'lambda_inversion_identity': lambda_inversion_identity,
        'lambda_stage2_perceptual': lambda_stage2_perceptual,
        'gradient_log_interval': gradient_log_interval,
        'pipeline': 'staged',
        'stage2_inject_latent_noise': stage2_latent_noise,
        'stage2_pixel_resolution': int(stage2_pixel_resolution),
        'save_progress_images': save_progress_images,
        'run_name': run_name,
    }
    with open(os.path.join(outdir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(run_config, f, indent=2)

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        network_data = legacy.load_network_pkl(f)
        G = network_data['G_ema'].to(device)  # type: ignore

    G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * sampling_multiplier)
    G.rendering_kwargs['depth_resolution_importance'] = int(
        G.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)
    if nrr is not None: G.neural_rendering_resolution = nrr

    image = Image.open(image_path).convert('RGB')
    c = np.load(c_path)
    c = np.reshape(c,(1,25))

    c = torch.FloatTensor(c).cuda()

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.Resize((512,512))
    ])
    from_im = trans(image).cuda()
    id_image = torch.squeeze((from_im.cuda() + 1) / 2) * 255

   
    deid_loss_fn = DeIDLoss(pp=pp)
    attr_loss_fn = AttrLoss()
    checkpoint_dir = os.path.join(outdir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    stage_metrics = []
    print(f'Stage 1/3: reconstructing input pivot ({num_steps_inversion} steps)...')
    torch.cuda.synchronize()
    stage_started = time.perf_counter()
    input_pivot = w_plus_editor.project(
        G, c, outdir, id_image, device=torch.device('cuda'),
        w_avg_samples=600, w_name=image_name, num_steps=num_steps_inversion,
        deid_loss=deid_loss_fn, attr_loss=attr_loss_fn,
        lamda_origin=1.0,
        lambda_deid=0.0, lambda_gender=0.0, lambda_expr=0.0,
        lambda_latent=0.0,
        stage_name='inversion', enable_privacy=False,
        optimize_noise=False,
        lambda_perceptual=lambda_inversion_perceptual,
        lambda_identity_preservation=lambda_inversion_identity,
        full_resolution_pixel=True,
        save_progress_images=save_progress_images)
    torch.cuda.synchronize()
    stage_seconds = time.perf_counter() - stage_started
    np.save(
        f'{checkpoint_dir}/{image_name}_input.npy',
        input_pivot.cpu().detach())
    w_avg = np.load('./w_avg.npy').astype(np.float32)
    inversion_reference = w_plus_editor.prepare_w_plus(
        w_avg, w_avg, G.backbone.mapping.num_ws, device)
    stage_metrics.append(w_plus_editor.save_stage_result(
        G, c, id_image, input_pivot, inversion_reference,
        deid_loss_fn, attr_loss_fn, outdir, 'inversion', stage_seconds,
        device))
    print(f'Stage 2/3: de-identifying from the reconstructed pivot ({num_steps} steps)...')

    torch.cuda.synchronize()
    stage_started = time.perf_counter()
    w_plus = w_plus_editor.project(
        G, c, outdir, id_image, device=torch.device('cuda'),
        w_avg_samples=600, w_name=image_name, num_steps=num_steps,
        initial_w=input_pivot,
        deid_loss=deid_loss_fn, attr_loss=attr_loss_fn,
        lamda_origin=lambda_origin,
        lambda_deid=lambda_deid,
        lambda_gender=lambda_gender, lambda_expr=lambda_expr,
        lambda_latent=lambda_latent,
        stage_name='deid',
        enable_privacy=True, optimize_noise=False,
        inject_latent_noise=stage2_latent_noise,
        lambda_perceptual=lambda_stage2_perceptual,
        full_resolution_pixel=(stage2_pixel_resolution == '512'),
        gradient_log_interval=gradient_log_interval,
        save_progress_images=save_progress_images)
    torch.cuda.synchronize()
    stage_seconds = time.perf_counter() - stage_started
    w_avg = np.load('./w_avg.npy').astype(np.float32)
    deid_reference = input_pivot
    deid_stage_name = 'deid'
    stage_metrics.append(w_plus_editor.save_stage_result(
        G, c, id_image, w_plus, deid_reference, deid_loss_fn, attr_loss_fn,
        outdir, deid_stage_name, stage_seconds, device))

    print(f'Stage 3/3: fine-tuning generator ({num_steps_pti} steps)...')
    torch.cuda.synchronize()
    stage_started = time.perf_counter()
    G_final = w_plus_editor.project_pti(
        G, c, outdir, id_image, w_plus, device=torch.device('cuda'),
        w_avg_samples=600, w_name=image_name, num_steps_pti=num_steps_pti,
        deid_loss=deid_loss_fn, attr_loss=attr_loss_fn,
        lamda_origin=lambda_origin,
        lambda_deid=lambda_deid,
        lambda_gender=lambda_gender, lambda_expr=lambda_expr,
        save_progress_images=save_progress_images)
    torch.cuda.synchronize()
    stage_seconds = time.perf_counter() - stage_started
    stage_metrics.append(w_plus_editor.save_stage_result(
        G_final, c, id_image, w_plus, deid_reference, deid_loss_fn,
        attr_loss_fn, outdir, 'post', stage_seconds, device))
    shutil.copy2(
        os.path.join(outdir, 'post', 'final.png'),
        os.path.join(outdir, 'final.png'))
    with open(os.path.join(outdir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump({'seed': seed, 'stages': stage_metrics}, f, indent=2)
    
    np.save(f'{checkpoint_dir}/{image_name}.npy', w_plus.cpu().detach())
    
    # Move every nn.Module in the dict to CPU before pickling so the checkpoint
    # can be deserialized on CPU-only nodes (e.g. the video generation task).
    network_data["G_ema"] = G_final.eval().requires_grad_(False).cpu()
    for k, v in network_data.items():
        if isinstance(v, torch.nn.Module):
            network_data[k] = v.eval().requires_grad_(False).cpu()
    with open(f'{checkpoint_dir}/fintuned_generator.pkl', 'wb') as f:
        pickle.dump(network_data, f)
    
    
    # PTI_embedding_dir = f'./projector/PTI/embeddings/{image_name}'
    # os.makedirs(PTI_embedding_dir,exist_ok=True)

    #np.save(f'./projector/PTI/embeddings/{image_name}/{image_name}_{latent_space_type}.npy', w)

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
