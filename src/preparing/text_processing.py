import random
import pandas as pd

# Define templates globally so they are easy to edit
TEMPLATES = [
    "a photo of {a} {t}",
    "a high-quality image of {a} {t}",
    "a professional shot of the {a} {t}",
    "a {a} {t} shown in a studio setting",
    "centered view of a {a} {t}",
    "an image featuring a {a} {t}",
    "this is a {a} {t}",
    "close-up of a {a} {t}"
]

def clean_metadata(df):
    """
    Fills missing titles with 'name' and creates the 'prompt' column.
    """
    # 1. Fallback for title
    # If cleaned_title is null, use 'name'. If both null, use 'object'
    df['final_title'] = df['cleaned_title'].fillna(df['name']).fillna("object")
    
    # 2. Generate Prompt
    df['prompt'] = df.apply(generate_varied_prompt, axis=1)
    
    return df

def generate_varied_prompt(row):
    title = str(row['final_title']).lower().strip()
    
    # Collect attributes if they exist and aren't in the title
    attributes = []
    for col in ['style', 'color', 'material']:
        val = str(row.get(col, '')).lower().strip()
        if val not in ['nan', 'none', '', 'unknown'] and val not in title:
            attributes.append(val)
            
    attr_string = " ".join(attributes)
    
    # Select template
    template = random.choice(TEMPLATES)
    
    # Format and clean up double spaces
    prompt = template.format(a=attr_string, t=title)
    prompt = " ".join(prompt.split())
    
    return prompt