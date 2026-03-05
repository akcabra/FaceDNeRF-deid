"""Example script for running face de-identification with FaceDNeRF-deid.

Uses the CPP-DeID loss formulation (Meden et al., 2023):
    L = λ_PX·L_VGG + λ_LT·L_LT + λ_ID·L_ID(pp) + λ_GD·L_GD + λ_EX·L_EX

Paper default weights: λ_PX=1.0, λ_LT=0.0016, λ_ID=2.5, λ_GD=0.01, λ_EX=0.01

Requires 2× NVIDIA GPUs (3090/4090 or better) for PTI phase.

Required checkpoints in ./networks/:
    - ffhqrebalanced512-128.pkl   (EG3D generator)
    - model_ir_se50.pth           (ArcFace IR-SE-50 for identity loss)
    - gender_classifier.pth       (ResNet-18 gender classifier, train with train_gender_classifier.py)
    - affecnet8_epoch5_acc0.6209.pth  (DAN expression classifier from https://github.com/yaoing/DAN)
    - vgg16.pt                    (VGG-16 for perceptual loss)
    - trained_model_03.t7         (DPR relight model, already included)
"""

import time
import os

gpu_id1 = 0
gpu_id2 = 1
num_steps = 1000
num_steps_pti = 400

#### De-identification configurations ####
# pp (privacy parameter): hinge threshold on squared ArcFace distance.
# ArcFace features are L2-normed → ||f1-f2||² ∈ [0, 4], pp ∈ [0, 1] is practical.
# Lower pp → stronger de-identification (pp=0 always pushes identity apart).
# Paper default weights: lamda_deid=2.5, lamda_origin=1.0, lamda_gender=0.01,
#                        lamda_expr=0.01, lamda_latent=0.0016

image_ids = ["00018"]
network_path = "./networks/ffhqrebalanced512-128.pkl"

deid_configs = [
    # Strong de-identification (pp=0.0, always pushes identity apart)
    {"pp": 0.0, "lamda_deid": 2.5, "lamda_origin": 1.0,
     "lamda_gender": 0.01, "lamda_expr": 0.01, "lamda_latent": 0.0016},
    # Medium de-identification (pp=0.5, stop once distance > 0.5)
    {"pp": 0.5, "lamda_deid": 2.5, "lamda_origin": 1.0,
     "lamda_gender": 0.01, "lamda_expr": 0.01, "lamda_latent": 0.0016},
    # Weak de-identification (pp=1.0, stop once distance > 1.0)
    {"pp": 1.0, "lamda_deid": 2.5, "lamda_origin": 1.0,
     "lamda_gender": 0.01, "lamda_expr": 0.01, "lamda_latent": 0.0016},
]

output_dir = "./output_deid/"

#####
for image_id in image_ids:
    for i, cfg in enumerate(deid_configs):
        print("=" * 60)
        print(f"Start: {time.ctime()}")
        print(f"Image: {image_id}, Config {i}: pp={cfg['pp']}")

        command = (
            f"CUDA_VISIBLE_DEVICES={gpu_id1},{gpu_id2} "
            f"python run.py "
            f"--outdir='{output_dir}' "
            f"--network={network_path} "
            f"--sample_mult=2 "
            f"--image_path ./test_data/{image_id}.png "
            f"--c_path ./test_data/{image_id}.npy "
            f"--num_steps {num_steps} "
            f"--num_steps_pti {num_steps_pti} "
            f"--mode deid "
            f"--pp {cfg['pp']} "
            f"--lamda_deid {cfg['lamda_deid']} "
            f"--lamda_origin {cfg['lamda_origin']} "
            f"--lamda_gender {cfg['lamda_gender']} "
            f"--lamda_expr {cfg['lamda_expr']} "
            f"--lamda_latent {cfg['lamda_latent']}"
        )
        print(command)
        os.system(command)
        print(f"End: {time.ctime()}")

        ### Generate result video
        pp_str = cfg['pp']
        outdir_video = os.path.join(
            output_dir,
            f"{image_id}_deid_pp{pp_str}_{cfg['lamda_deid']}_{cfg['lamda_origin']}_{cfg['lamda_gender']}_{cfg['lamda_expr']}"
        )
        command = (
            f"CUDA_VISIBLE_DEVICES={gpu_id1} "
            f"python gen_videos_from_given_latent_code.py "
            f"--outdir='{outdir_video}' "
            f"--trunc=0.7 "
            f"--npy_path '{outdir_video}/checkpoints/{image_id}.npy' "
            f"--network='{outdir_video}/checkpoints/fintuned_generator.pkl' "
            f"--sample_mult=2"
        )
        print(command)
        os.system(command)
