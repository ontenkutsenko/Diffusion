# import
from transformers import AutoProcessor, AutoModel
import torch
from PIL import Image
from torchmetrics.multimodal.clip_score import CLIPScore
import numpy as np
from torchvision import transforms

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

# load model
device = "cuda:0" #TODO: add config file to configure device as env
processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"


processor = AutoProcessor.from_pretrained(processor_name_or_path)
model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)

def pick_score(prompt: str, 
               images: list[Image.Image]) -> list:
    
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
    
    return probs.cpu().tolist()

from torchmetrics.functional.multimodal import clip_score
from functools import partial

clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

def clip_score(images: list[Image.Image],
               prompts: list[str], ) -> list:
    images=np.array(images)
    images_int = (images * 255).astype("uint8")
    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)

def fid_score(real_images: list[Image.Image], 
             generated_images: list[Image.Image]) -> float:
    """Calculate the FID score between two sets of images"""

    fid = FrechetInceptionDistance(feature=2048).to(device)

    transform = transforms.Compose([
        transforms.Resize((299, 299)), 
        transforms.ToTensor()
    ])  

    real_images = torch.stack([transform(image.convert("RGB")).mul(255).to(torch.uint8) for image in real_images]).to(device)
    fid.update(real_images, real=True)

    generated_images = torch.stack([transform(image.convert("RGB")).mul(255).to(torch.uint8) for image in generated_images]).to(device)
    fid.update(generated_images, real=False)

    fid_score = fid.compute().item()
    return fid_score

def inception_score(images: list[Image.Image]) -> float:
    """Calculate the Inception Score of a set of images"""
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])  

    images = torch.stack([transform(image.convert("RGB")).mul(255).to(torch.uint8) for image in images]).to(device)
    
    inception = InceptionScore().to(device)
    inception.update(images)
    return inception.compute()[0].item()