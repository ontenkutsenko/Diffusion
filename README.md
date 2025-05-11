# Diffusion Sampling Efficiency – Retrieval-based Improvements
This project explores **sampling efficiency in Latent Diffusion Models** using retrieval-based techniques. It builds on top of [ReDi](https://arxiv.org/pdf/2302.02285) method and is part of the Master's thesis *"Enhancing the Efficiency of Diffusion Models"*.

![Demo with compressed keys](/src/data/results/demo_with_latents.png)

## 🔍 Project Goals

- Implement and evaluate **retrieval-based diffusion** with Retrieval Key Compression (PCA and PQ) to optimize memory usage and lookup speed for stored latent vectors.
- Implement an **adaptive skipping technique** for retrieval-based diffusion that adjusts the number of skipped denoising steps based on retrieval confidence.
- Evaluate performance using **Domain-Specific Knowledge Bases**, including:
  - **ID-2K**: a synthetic interior design dataset
  - **COCO-10K**: a subset of MS-COCO filtered for furniture-related prompts
- Benchmark improvements over standard DDIM/Stable Diffusion baselines using:
  - **CLIPScore** (text-image alignment)
  - **PickScore** (human preference modeling)
  - **FID** (distribution similarity)
  - **Inception Score (IS)** (image diversity and confidence)
- [WIP] Integrate a **Value Refinement Module**: an auxiliary lightweight U-Net that refines retrieved latents to improve quality in the later denoising steps.

## 📂 Structure

- Prompts preparation for [`ID-2K`](/src/stable/prompt.ipynb) and [`COCO-10K`](/src/stable/coco_furniture_images.ipynb)
- [`Main pipeline`](/src/notebooks/whole%20pipeline.ipynb): Trajectories generation, experiments with key compression and demos
- [`Adaptive skipping`](/src/notebooks/adaptive_skipping.ipynb): Retrieval logic and compression tools
- [`Retrieval Diffusion module`](/src/retrieval/): All functions to work with Retrieval Diffusion (Knowledge Base, Trajectories, Generation, Neighbours search). Partially based on ReDi approach.
- [`Results`](/src/data/results/): All the artifacts of experimentation
- [`Data`](/src/data/): Lightweight files with prompts etc. Main datsets with trajectories are kept in [HuggingFace dataset](https://huggingface.co/datasets/ontenkutsenko/diffusion_trajectories)
- [`Metrics`](/src/metrics/): Functions for computing FID, CLIPScore, PickScore and IS

## ✅ Requirements

Dependencies installation and virtual environment creation is managed by PDM

Install dependencies:
```bash
pdm install
```