import webdataset as wds
import os
import shutil
from src.preparing.image_processing import process_image, image_to_bytes
from tqdm import tqdm

def write_dataset(df, output_folder, image_root_dir, config):
    """
    Iterates through the DataFrame and writes WebDataset shards.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    shard_name = "shard-%06d.tar"
    pattern = "file:" + os.path.join(output_folder, shard_name).replace("\\", "/")
    
    print(f"Writing {len(df)} samples to {output_folder}...")
    
    with wds.ShardWriter(pattern, maxsize=config.MAX_SHARD_SIZE, maxcount=config.MAX_COUNT_PER_SHARD) as sink:
        for index, row in tqdm(df.iterrows(), total=len(df)):
            full_path = os.path.join(image_root_dir, row['local_path'])
            
            img = process_image(
                full_path, 
                target_size=config.IMAGE_SIZE, 
                fill_color=config.PADDING_COLOR
            )
            
            if img is None:
                continue
                
            sink.write({
                "__key__": f"{index:09d}",
                "jpg": image_to_bytes(img, config.IMAGE_QUALITY),
                "txt": row['prompt'],
                "json": {
                    "original_title": row['final_title'],
                    "style": str(row['style']),
                    "color": str(row['color']),
                    "material": str(row['material'])
                }
            })
            
    print(f"Finished writing {output_folder}")

def create_mini_train(train_folder, mini_folder, num_shards):
    """
    Copies the first N shards from Train to a Mini-Train folder.
    """
    os.makedirs(mini_folder, exist_ok=True)
    files = sorted([f for f in os.listdir(train_folder) if f.endswith('.tar')])
    
    # Take top N
    files_to_copy = files[:num_shards]
    
    print(f"Creating Mini-Train: Copying {len(files_to_copy)} shards...")
    for f in files_to_copy:
        src = os.path.join(train_folder, f)
        dst = os.path.join(mini_folder, f)
        shutil.copy2(src, dst)
    print("Mini-Train created.")