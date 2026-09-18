# CPP-DeID-inspired de-identification method with EG3D

This repository contains a proof-of-concept method for controllable face de-identification with EG3D inspired by CPP-DeID. Given an EG3D-compatible face image and camera vector, the method:

1. reconstructs the input in EG3D's W+ latent space;
2. optimizes the reconstruction toward a requested ArcFace similarity (`pp`);
3. preserves image structure, gender, and expression with auxiliary losses;
4. optionally renders the de-identified identity from additional viewpoints.

Lower `pp` values request a larger identity change. The implementation supports the CPP-DeID-inspired loss profile used in our evaluation and records the run configuration and stage metrics for reproducibility.

## Origin of the code

This project is a direct fork of [FaceDNeRF](https://github.com/BillyXYB/FaceDNeRF), which itself builds on [EG3D](https://github.com/NVlabs/eg3d). A substantial part of the repository is therefore inherited code and was not written or used directly for this project. In particular, the EG3D generator, rendering infrastructure, W+ projection, pivotal tuning, and supporting modules originate from FaceDNeRF/EG3D.

Our contribution is the staged de-identification pipeline, CPP-DeID-inspired losses, run tracking, independent evaluation, controllable input handling, and multi-view rendering/evaluation support. Unused original command-line utilities are retained under `facednerf_tools/` for provenance; they are not part of the reported workflow.

## Repository layout

```text
run.py                 Main reconstruction and de-identification entry point
bash_scripts/          Slurm launch scripts
criteria/              Identity, attribute, and preservation losses
editors/               W+ optimization and pivotal tuning
evaluation_models/     ArcFace and independent FaceNet wrappers
scripts/               Evaluation, preprocessing, and multi-view tools
evaluation/results/    Compact CSV results used in the report
facednerf_tools/       Unused utilities inherited from FaceDNeRF/EG3D
training/, torch_utils/,
dnnlib/, legacy.py     Inherited EG3D runtime required to load and render models
```

## Setup

Create the provided Conda environment:

```bash
conda env create -f environment.yml
conda activate facednerf
```

CUDA compilation of the EG3D operators requires a CUDA toolkit and compatible
host compiler. The evaluation runs used Python 3.9, PyTorch 1.11, and CUDA 11.3.

### Required model files

Create `networks/` in the repository root and place the following files in it:

```text
networks/
├── ffhqrebalanced512-128.pkl       Pretrained EG3D FFHQ generator
├── vgg16.pt                        StyleGAN VGG feature extractor
├── model_ir_se50.pth               ArcFace identity model
├── gender_classifier.pth           CelebA-trained gender classifier
├── affecnet8_epoch5_acc0.6209.pth  DAN expression classifier
└── 20180402-114759-vggface2.pt     FaceNet model used only for evaluation
```

### Input format

Each input must be an EG3D-compatible pair with the same filename stem:

```text
inputs/
├── 00018.png   Aligned 512x512 RGB face image
└── 00018.npy   Matching 25-dimensional EG3D camera vector
```

The image and camera vector must come from the same EG3D preprocessing result. Arbitrarily resizing or cropping an image without recomputing its camera vector causes poor inversion. Dataset files should be stored under `data/` or another local directory and should not be committed.

## Running de-identification

On a Slurm cluster, run one configuration with:

```bash
sbatch --export=ALL,IMAGE_ID=00018,INPUT_DIR=./inputs,PP=0.3,OUTDIR=./output_deid \
  bash_scripts/slurm_deid.sh
```

The evaluation configuration defaults to 300 inversion steps, 500 de-identification steps, 400 pivotal-tuning steps, seed 42, and final images only. Parameters can be overridden through `sbatch --export`, including `NUM_STEPS`, `NUM_STEPS_INVERSION`, `NUM_STEPS_PTI`, `SEED`, and the loss coefficients defined in `bash_scripts/slurm_deid.sh`.

Every run receives its own directory and contains:

```text
final.png             Final de-identified input-view image
config.json           Complete run configuration
metrics.json          Metrics and runtime for each stage
inversion/final.png   Final Stage 1 reconstruction
deid/final.png        Final W+ de-identification render
post/final.png        Final render after pivotal tuning
checkpoints/          Optimized latent and fine-tuned generator
```

Generated outputs and checkpoints can be very large and should not be committed.

## Multi-view rendering

Render fixed yaw views from one completed run with:

```bash
python scripts/render_multiview.py \
  --run-dir PATH_TO_RUN \
  --output-dir PATH_TO_OUTPUT
```

The renderer uses the saved W+ representation and fine-tuned generator while
changing only the EG3D camera.
