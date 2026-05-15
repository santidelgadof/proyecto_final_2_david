# Experiment Results

These results were obtained on the provided Cityscapes subset using the fixed split generated with seed `42`:

- Train: 350 images
- Validation: 75 images
- Test: 75 images

## Baseline

Configuration:

- Generator: U-Net Pix2Pix
- Discriminator: PatchGAN
- Loss: BCE adversarial + `100 * L1`
- Data augmentation: no
- Epochs: 3
- Batch size: 4

Test metrics:

- MAE: `0.2006`
- PSNR: `17.0483`
- SSIM: `0.6023`

Artifacts:

- Metrics summary: `runs/baseline/summary.json`
- Example inference: `outputs/image_001_baseline.png`

## Improved model

Configuration:

- Generator: U-Net Pix2Pix with 2 residual bottleneck blocks
- Discriminator: PatchGAN
- Loss: LSGAN adversarial + `100 * L1`
- Data augmentation: paired horizontal flip + resize/crop with 16-pixel margin
- Epochs: 3
- Batch size: 4

Test metrics:

- MAE: `0.1982`
- PSNR: `17.0078`
- SSIM: `0.6100`

Artifacts:

- Checkpoint for demo inference: `runs/improved/infer.pt`
- Summary: `runs/improved/summary.json`
- Example inference: `outputs/image_001_improved.png`

## Comparison

Compared with the baseline, the improved model achieved:

- Better MAE: `0.2006 -> 0.1982`
- Better SSIM: `0.6023 -> 0.6100`
- Similar PSNR: `17.0483 -> 17.0078`

At 3 epochs the improvement is already visible in structural similarity. Additional epochs are the clearest next lever if a stronger final report is needed.