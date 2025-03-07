import torch
from diffusers.pipelines import StableDiffusionPipeline
import json
import h5py
import xxhash
import matplotlib.pyplot as plt

def generate_trajectory(prompt: str, 
                        pipeline: StableDiffusionPipeline,
                        device: str = "cpu") -> tuple:
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
    ).images

    #show image
    plt.imshow(images[0])
    plt.show()

    return (trajectory, prompt)

def save_trajectory(trajectory: str, 
                    prompt: str,
                    trajectory_filename: str,
                    prompt_filename: str):
    dataset_name = xxhash.xxh32(prompt.encode("utf-8")).hexdigest()

    with open(f"{prompt_filename}.jsonl", "a") as f:
        row = {dataset_name: prompt}
        f.write(json.dumps(row) + "\n")

    with h5py.File(f"{trajectory_filename}.h5", "a") as hf:
        if dataset_name in hf:  # Check if dataset already exists
            print(f"Warning: Dataset {dataset_name} already exists! Skipping write.")
        else:
            hf.create_dataset(dataset_name, 
                            data=trajectory, 
                            compression="gzip", 
                            compression_opts=4)
            
def generate_trajectory_from_latents():
    pass
