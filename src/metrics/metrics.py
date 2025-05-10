# import
import numpy as np
import torch
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchvision import transforms
from transformers import AutoModel, AutoProcessor

# load model
device = "cuda:1"  # TODO: add config file to configure device as env
processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"


processor = AutoProcessor.from_pretrained(processor_name_or_path)
model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)


def pick_score(prompt: str, images: list[Image.Image]) -> list:
    try:
        # preprocess
        image_inputs = processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)

        text_inputs = processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            # embed
            image_embs = model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

            text_embs = model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

            # score
            scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]

            # get probabilities if you have multiple images to choose from
            probs = torch.softmax(scores, dim=-1)
            
        del image_inputs, text_inputs, image_embs, text_embs, scores
        if torch.cuda.is_available():
            # clear GPU memory
            torch.cuda.empty_cache()

        return probs.cpu().tolist()
    finally:
        del probs


from functools import partial

from torchmetrics.functional.multimodal import clip_score

clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")


def clip_score(images: list[Image.Image], prompts: list[str], batch_size=64):

    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        scores = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                batch_prompts = prompts[i:i + batch_size]
                tensor = torch.stack([transform(img) for img in batch_images])
                tensor = (tensor * 255).to(torch.uint8).to(device)
                score = clip_score_fn(tensor, batch_prompts).detach().cpu()
                scores.append(score)

        # average all batch scores
        return round(torch.stack(scores).mean().item(), 4)
    finally:
        del transform, scores
        if torch.cuda.is_available():
            # clear GPU memory
            torch.cuda.empty_cache()

def compute_fid(real_images, gen_images, batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor()
    ])
    fid = FrechetInceptionDistance(feature=2048).to(device)

    def update(images, real_flag):
        for i in range(0, len(images), batch_size):
            batch = torch.stack([
                transform(img.convert("RGB")).mul(255).to(torch.uint8)
                for img in images[i:i+batch_size]
            ]).to(device)
            fid.update(batch, real=real_flag)
            del batch
            torch.cuda.empty_cache()

    with torch.no_grad():
        update(real_images, True)
        update(gen_images, False)

    return fid.compute().item()

def fid_score(
    real_images: list[Image.Image], generated_images: list[Image.Image],
    batch_size: int = 64
) -> float:
    """Calculate the FID score between two sets of images"""
    try:

        fid = FrechetInceptionDistance(feature=2048).to(device)

        transform = transforms.Compose(
            [transforms.Resize((299, 299)), transforms.ToTensor()]
        )

        def update(images, real_flag):
            for i in range(0, len(images), batch_size):
                batch = torch.stack([
                    transform(img.convert("RGB")).mul(255).to(torch.uint8)
                    for img in images[i:i+batch_size]
                ]).to(device)
                fid.update(batch, real=real_flag)
                del batch
                torch.cuda.empty_cache()

        with torch.no_grad():
            update(real_images, True)
            update(generated_images, False)

        return fid.compute().item()
    finally:
        del fid
        if torch.cuda.is_available():
            # clear GPU memory
            torch.cuda.empty_cache()


def inception_score(images: list[Image.Image], 
                    batch_size: int = 64) -> float:
    """Calculate the Inception Score of a set of images"""
    
    try:

        transform = transforms.Compose([transforms.ToTensor()])

        inception = InceptionScore().to(device)
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = torch.stack([
                    transform(img.convert("RGB")).mul(255).to(torch.uint8)
                    for img in images[i:i+batch_size]
                ]).to(device)
                inception.update(batch)
                del batch
                torch.cuda.empty_cache()
        return inception.compute()[0].item()
    finally:
        del inception
        if torch.cuda.is_available():
            # clear GPU memory
            torch.cuda.empty_cache()
