# Image-to-Image Translation with Pix2Pix

**Authors:** Santiago Delgado Ferreiro and David Carballo Rodriguez

This repository implements the final Computer Vision II project for paired image-to-image translation on the assigned Cityscapes subset. The code includes:

- A baseline Pix2Pix experiment.
- An improved variant with paired augmentation, residual bottleneck blocks, and LSGAN loss.
- Evaluation with MAE, PSNR, and SSIM.
- A CLI demo for inference from a paired sample or a standalone label map.

## Project structure

- `cityscapes/`: extracted paired dataset.
- `src/pix2pix_project/data.py`: dataset reading and reproducible train/val/test splits.
- `src/pix2pix_project/models.py`: U-Net generator and PatchGAN discriminator.
- `src/pix2pix_project/train.py`: training, validation, checkpointing, metrics, and preview generation.
- `src/pix2pix_project/infer.py`: model loading and inference CLI.
- `train.py`: root entrypoint for training.
- `demo.py`: root entrypoint for inference.

## Dataset format

The dataset is expected to contain paired images where:

- Left half: real RGB image.
- Right half: label map.

The training task is `label map -> realistic image`.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Training

Baseline:

```bash
python train.py --preset baseline --data-dir cityscapes --output-dir runs/baseline
```

Improved model:

```bash
python train.py --preset improved --data-dir cityscapes --output-dir runs/improved
```

Useful fast smoke test:

```bash
python train.py --preset baseline --data-dir cityscapes --epochs 1 --max-train-batches 2 --max-eval-batches 1
```

## Inference demo

Inference from a paired dataset image:

```bash
python demo.py --checkpoint runs/improved/infer.pt --input cityscapes/image_001.jpg --output outputs/image_001_pred.png --input-kind paired
```

Inference from a standalone label map image:

```bash
python demo.py --checkpoint runs/improved/infer.pt --input sample_label.png --output outputs/sample_pred.png --input-kind label
```

The published repository keeps a single lightweight inference checkpoint (`runs/improved/infer.pt`) plus metrics and previews for both experiments.

## Outputs

Each training run stores:

- `config.json`: exact experiment configuration.
- `history.json`: train/validation metrics by epoch.
- `infer.pt`: inference-only checkpoint (generator + config).
- `summary.json`: final run summary and test metrics.
- `previews/`: generated visual samples for validation and testing.

## Report

The academic report is provided in `report.tex` with BibTeX references in `references.bib` and the qualitative figure in `figures/qualitative_comparison.png`.

To compile it in a LaTeX environment with IEEE support:

```bash
pdflatex report.tex
bibtex report
pdflatex report.tex
pdflatex report.tex
```

## Suggested report angle

- Baseline: original Pix2Pix objective with BCE adversarial loss and L1 reconstruction.
- Improved version: stronger generator bottleneck, LSGAN loss, and paired augmentation.
- Compare MAE, PSNR, and SSIM on the same fixed split.