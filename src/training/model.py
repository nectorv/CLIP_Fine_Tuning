from transformers import CLIPModel, CLIPProcessor
from peft import LoraConfig, get_peft_model
import torch.nn as nn

def get_model(scenario, lora_r=16, lora_alpha=32, lora_dropout=0.05):
    print(f"🏗️ Building model for scenario: {scenario}")
    
    # Load base model
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)

    # Freeze everything by default first
    for param in model.parameters():
        param.requires_grad = False

    if scenario == "zero_shot":
        return model, processor # Everything frozen

    elif scenario == "linear_probe":
        # Unfreeze ONLY projections
        for param in model.visual_projection.parameters():
            param.requires_grad = True
        for param in model.text_projection.parameters():
            param.requires_grad = True
        print("🔓 Unfrozen: Projections only")

    elif scenario == "dual_lora":
        # Unfreeze projections fully
        for param in model.visual_projection.parameters():
            param.requires_grad = True
        for param in model.text_projection.parameters():
            param.requires_grad = True
            
        # Apply LoRA to Backbones
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"], # Targets Attention in both ViT and Text
            lora_dropout=lora_dropout,
            bias="none"
        )
        model = get_peft_model(model, config)
        print("🔓 Applied LoRA to encoders + Unfrozen Projections")

    # Add logic for vision_lora / text_lora similarly if needed

    return model, processor