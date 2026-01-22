import pandas as pd
import os
from sklearn.model_selection import train_test_split
from src.config import PrepConfig
from src.preparing.text_processing import clean_metadata
from src.preparing.sharding import write_dataset, create_mini_train

def main():
    # 1. Load Data
    print(f"Loading metadata from {PrepConfig.INPUT_CSV_PATH}...")
    df = pd.read_csv(PrepConfig.INPUT_CSV_PATH)
    df = df.sample(frac=1, random_state=PrepConfig.SEED).reset_index(drop=True)
    df = df.dropna(subset=['local_path'])
    
    # 2. Clean & Preprocess Text
    print("Preprocessing text and generating prompts...")
    df = clean_metadata(df)
    
    # 3. Split Data
    # First split: Train vs (Temp = Val + Test)
    train_df, temp_df = train_test_split(
        df, 
        test_size=(PrepConfig.VAL_RATIO + PrepConfig.TEST_RATIO), 
        random_state=PrepConfig.SEED,
        shuffle=True
    )
    
    # Second split: Val vs Test
    # Adjust ratio because temp_df is smaller than original df
    relative_test_ratio = PrepConfig.TEST_RATIO / (PrepConfig.VAL_RATIO + PrepConfig.TEST_RATIO)
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=relative_test_ratio, 
        random_state=PrepConfig.SEED,
        shuffle=True
    )
    
    print(f"Data Split Summary:")
    print(f"Train: {len(train_df)}")
    print(f"Val:   {len(val_df)}")
    print(f"Test:  {len(test_df)}")
    
    # 4. Write Datasets (Sharding)
    # Train
    write_dataset(
        train_df, 
        os.path.join(PrepConfig.OUTPUT_DIR, "train"), 
        PrepConfig.IMAGES_ROOT_DIR, 
        PrepConfig
    )
    
    # Validation
    write_dataset(
        val_df, 
        os.path.join(PrepConfig.OUTPUT_DIR, "validation"), 
        PrepConfig.IMAGES_ROOT_DIR, 
        PrepConfig
    )
    
    # Test
    write_dataset(
        test_df, 
        os.path.join(PrepConfig.OUTPUT_DIR, "test"), 
        PrepConfig.IMAGES_ROOT_DIR, 
        PrepConfig
    )
    
    # 5. Create Mini-Train Subset
    create_mini_train(
        os.path.join(PrepConfig.OUTPUT_DIR, "train"),
        os.path.join(PrepConfig.OUTPUT_DIR, "mini_train"),
        PrepConfig.MINI_TRAIN_SHARD_COUNT
    )
    
    print("\nAll tasks completed successfully!")
    print(f"Ready for AWS S3 Upload: {PrepConfig.OUTPUT_DIR}")

if __name__ == "__main__":
    main()