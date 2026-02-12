from transformers import CLIPModel, CLIPProcessor
from peft import LoraConfig, get_peft_model
import torch.nn as nn

from src.config import TrainingRunConfig

def get_model(scenario, lora_r=16, lora_alpha=32, lora_dropout=0.05):
    print(f"🏗️ Building model for scenario: {scenario}")
    
    # Load base model
    model = CLIPModel.from_pretrained(TrainingRunConfig.MODEL_ID)
    processor = CLIPProcessor.from_pretrained(TrainingRunConfig.MODEL_ID)

    # Freeze everything by default first
    for param in model.parameters():
        param.requires_grad = False

    if scenario == "zero_shot":
        return model, processor # Everything frozen

    elif scenario == "linear_probe":
        # Apply LoRA to projections only
        projection_targets = ["visual_projection", "text_projection"]
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=projection_targets,
            lora_dropout=lora_dropout,
            bias="none"
        )
        model = get_peft_model(model, config)
        print("🔓 Applied LoRA: Projections only")

    elif scenario == "dual_lora":
        # Apply LoRA to backbones + projections
        projection_targets = ["visual_projection", "text_projection"]
        target_modules = TrainingRunConfig.LORA_TARGET_MODULES + projection_targets
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules, # Targets Attention in both ViT and Text + projections
            lora_dropout=lora_dropout,
            bias="none"
        )
        model = get_peft_model(model, config)
        print("🔓 Applied LoRA to encoders + projections")

    elif scenario == "unfrozen_targets":
        # Unfreeze target modules (no LoRA)
        projection_targets = ["visual_projection", "text_projection"]
        target_modules = TrainingRunConfig.LORA_TARGET_MODULES + projection_targets
        for name, param in model.named_parameters():
            if any(target in name for target in target_modules):
                param.requires_grad = True
        print("🔓 Unfroze target modules: encoders + projections")

    # Add logic for vision_lora / text_lora similarly if needed

    return model, processor