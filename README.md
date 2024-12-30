<h1 align="center">✨ 
<br/>  
TangoFlux: Super Fast and Faithful Text to Audio Generation with Flow Matching and Clap-Ranked Preference Optimization 
<br/>
✨✨✨


</h1>

<div align="center">
  <img src="assests/tf_teaser.png" alt="TangoFlux" width="1000" />

<br/>

[![arXiv](https://img.shields.io/badge/Read_the_Paper-blue?link=https%3A%2F%2Fopenreview.net%2Fattachment%3Fid%3DtpJPlFTyxd%26name%3Dpdf)
](https://openreview.net/attachment?id=tpJPlFTyxd&name=pdf) ![Static Badge](https://img.shields.io/badge/TangoFlux-Huggingface-violet?logo=huggingface&link=https%3A%2F%2Fhuggingface.co%2Fdeclare-lab%2FTangoFlux) [![Static Badge](https://img.shields.io/badge/Demos-declare--lab-brightred?style=flat)](https://tangoflux.github.io/) ![Static Badge](https://img.shields.io/badge/TangoFlux-Huggingface_Space-8A2BE2?logo=huggingface&link=https%3A%2F%2Fhuggingface.co%2Fspaces%2Fdeclare-lab%2FTangoFlux) ![Static Badge](https://img.shields.io/badge/TangoFlux_Dataset-Huggingface-red?logo=huggingface&link=https%3A%2F%2Fhuggingface.co%2Fdatasets%2Fdeclare-lab%2FTangoFlux)




</div>

## Overall Pipeline
TangoFlux consists of FluxTransformer blocks which are Diffusion Transformer (DiT) and Multimodal Diffusion Transformer (MMDiT), conditioned on textual prompt and duration embedding to generate audio at 44.1kHz up to 30 seconds. TangoFlux learns a rectified flow trajectory from audio latent representation encoded by a variational autoencoder (VAE). The TangoFlux training pipeline consists of three stages: pre-training, fine-tuning, and preference optimization. TangoFlux is aligned via CRPO which iteratively generates new synthetic data and constructs preference pairs to perform preference optimization.

![cover-photo](assests/tangoflux.png)

## Quickstart

## Training TangoFlux

## Inference with TangoFlux

## Evaluation Scripts

## Comparison Between TangoFlux and Other Audio Generation Models

This comparison evaluates TangoFlux and other audio generation models across various metrics. Key metrics include:

- **Output Length**: Represents the duration of the generated audio.
- **FD**<sub>openl3</sub>: Frechet Distance.
- **KL**<sub>passt</sub>: KL divergence.
- **CLAP**<sub>score</sub>: Alignment score.


All inference times are computed on the same A40 GPU. The trainable parameters are reported in the **\#Params** column.

| Model                           | \#Params  | Duration | Steps | FD<sub>openl3</sub> ↓ | KL<sub>passt</sub> ↓ | CLAP<sub>score</sub> ↑ | IS ↑ | Inference Time (s) |
|---------------------------------|-----------|----------|-------|-----------------------|----------------------|------------------------|------|--------------------|
| **AudioLDM 2-large**            | 712M      | 10 sec   | 200   | 108.3                | 1.81                 | 0.419                  | 7.9  | 24.8               |
| **Stable Audio Open**           | 1056M     | 47 sec   | 100   | 89.2                 | 2.58                 | 0.291                  | 9.9  | 8.6                |
| **Tango 2**                     | 866M      | 10 sec   | 200   | 108.4                | **1.11**             | 0.447                  | 9.0  | 22.8               |
| **TangoFlux-base**              | **515M**  | 30 sec   | 50    | 80.2                 | 1.22                 | 0.431                  | 11.7 | **3.7**            |
| **TangoFlux**                   | **515M**  | 30 sec   | 50    | **75.1**             | 1.15                 | **0.480**              | **12.2** | **3.7**         |



## Citation

```bibtex

@article{Hung2025TangoFlux,
  title = {TangoFlux: Super Fast and Faithful Text to Audio Generation with Flow Matching and Clap-Ranked Preference Optimization},
  author = {Chia-Yu Hung and Navonil Majumder and Zhifeng Kong and Ambuj Mehrish and Rafael Valle and Bryan Catanzaro and Soujanya Poria},
  year = {2025},
  url = {https://openreview.net/attachment?id=tpJPlFTyxd&name=pdf},
  note = {Available at OpenReview}
}

```
