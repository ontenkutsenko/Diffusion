import json
import time

import faiss
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import xxhash
from diffusers import SchedulerMixin, StableDiffusionPipeline

from redi.pipeline_re_sd import ReSDPipeline


def generate_trajectory(
    prompt: str,
    pipeline: StableDiffusionPipeline,
    scheduler: SchedulerMixin,
    num_inference_steps: int = 50,
    device: str = "cpu",
) -> tuple:
    pipeline.scheduler = scheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)
    generator = torch.Generator(device).manual_seed(1024)

    def collect_latents(step, timestep, latents, trajectory):
        trajectory.append(latents.cpu().numpy())

    trajectory = []
    images = pipeline(
        prompt,
        callback_steps=1,
        callback=lambda step, timestep, latents: collect_latents(
            step, timestep, latents, trajectory
        ),
        guidance_scale=7.5,
        generator=generator,
        num_inference_steps=num_inference_steps,
    ).images

    # show image
    plt.imshow(images[0])
    plt.show()
    del images

    return (trajectory, prompt)


def save_trajectory(
    trajectory: str, prompt: str, trajectory_filename: str, prompt_filename: str
):
    dataset_name = xxhash.xxh32(prompt.encode("utf-8")).hexdigest()

    with open(f"{prompt_filename}.jsonl", "a") as f:
        row = {dataset_name: prompt}
        f.write(json.dumps(row) + "\n")

    with h5py.File(f"{trajectory_filename}.h5", "a") as hf:
        if dataset_name in hf:  # Check if dataset already exists
            print(f"Warning: Dataset {dataset_name} already exists! Skipping write.")
        else:
            hf.create_dataset(
                dataset_name, data=trajectory, compression="gzip", compression_opts=4
            )


def generate_trajectory_from_latents(
    prompt: str,
    latent: np.ndarray,
    pipeline: ReSDPipeline,
    scheduler: SchedulerMixin,
    num_inference_steps: int = 30,
    value_margin_steps: int = 10,
    device: str = "cpu",
) -> torch.Tensor:
    pipeline = pipeline.to(device)
    # generator = torch.Generator(device).manual_seed(1024)
    # img = pipeline(
    #     prompt,
    #     head_start_latents=torch.tensor(latent).to(device),
    #     head_start_step=num_inference_steps - value_margin_steps,
    #     guidance_scale=7.5,
    #     generator=generator,
    #     num_inference_steps=num_inference_steps,
    #     scheduler = scheduler
    # ).images[0]
    img = pipeline.generate_from_latent(
        prompt=prompt,
        latent=torch.tensor(latent).to(device),
        head_start_step=num_inference_steps - value_margin_steps,
        guidance_scale=7.5,
        # generator=generator,
        num_inference_steps=num_inference_steps,
        scheduler=scheduler,
    )

    return img


# def generate_from_latent(
#         self,
#         latent: torch.FloatTensor,
#         prompt: str,
#         num_inference_steps: int = 30,
#         head_start_step: int = 0,
#         eta: float = 0.0,
#         guidance_scale: float = 7.5,
#         output_type: str = "pil",
#         scheduler: SchedulerMixin | None = None,
#         device: str = "cpu",
#     ):


def retrieve_nearest_neigbours(
    query_array,
    num_neighbours: int = 1,
    index_path: str = "faiss_index.bin",
    kb_path: str = "knowledge_base.h5",
) -> list[tuple[np.ndarray, float, str]]:
    """Retrieve the closest stored array using FAISS"""
    query_flat = query_array.flatten().reshape(1, -1)
    # TODOD: read index outside the function
    index = faiss.read_index(index_path)

    # Perform FAISS search
    start = time.time()
    D, I = index.search(query_flat, num_neighbours)
    end = time.time()
    nearest_ids = I[0]

    with h5py.File(kb_path, "r") as kb_file:
        neighbours = []
        for i in range(len(nearest_ids)):
            nearest_id_str = str(nearest_ids[i])
            stored_array = kb_file[nearest_id_str][()]
            neighbours.append((stored_array, D[0][i], nearest_id_str))
    return neighbours, end - start
