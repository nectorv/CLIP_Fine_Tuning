from PIL import Image, ImageOps
import io

def process_image(image_path, target_size=224, fill_color=(255, 255, 255)):
    """
    Opens an image, converts to RGB, and applies letterbox resizing 
    (preserving aspect ratio) with padding.
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            
            # Calculate aspect ratio preserving resize
            old_size = img.size
            ratio = float(target_size) / max(old_size)
            new_size = tuple([int(x * ratio) for x in old_size])
            
            # Resize
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Create new canvas
            new_img = Image.new("RGB", (target_size, target_size), fill_color)
            
            # Paste in center
            # (target - new) // 2 calculates the offset to center it
            paste_pos = ((target_size - new_size[0]) // 2,
                         (target_size - new_size[1]) // 2)
            new_img.paste(img, paste_pos)
            
            return new_img
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def image_to_bytes(img, quality=95):
    """Converts PIL image to JPEG bytes for WebDataset storage."""
    if img is None: 
        return None
    with io.BytesIO() as output:
        img.save(output, format="JPEG", quality=quality)
        return output.getvalue()