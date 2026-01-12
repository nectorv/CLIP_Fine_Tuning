"""
System prompts for GPT-5 nano VLM data cleaning.

This module contains the prompt templates for vision-language model cleaning,
where the model analyzes both the product image and title to extract
structured information.
"""


def get_cleaning_prompt(raw_title: str) -> str:
    """
    Generate a user prompt for cleaning a product with image and title.
    
    Args:
        raw_title: The original noisy product title
        
    Returns:
        User prompt string for GPT-5 nano
    """
    return f"""
TASK :
Extract structured metadata from the image and Raw Title: "{raw_title}"

RULES :
1. CLEANING: Remove all brand name, SKUs, prices, shipping info, and 'Set of X' mentions.
2. VISUAL OVERRIDE: If the title says "Red" but the image is "Blue", use "Blue".

OUTPUT :
Return ONLY a valid JSON object.
{{
  "clean_title": "string",
  "style": "string",
  "material": "string",
  "color": "string",
  "object_type": "string"
}}
"""
