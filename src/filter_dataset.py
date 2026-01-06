# Steam Review Dataset Filtering and Preparation
# Group 150: Mustakim Kazi, Louis Doan, Michael Fedotov

import pandas as pd
import re

# FilePaths
INPUT_FILE = 'Datasets/all_reviews_100m.csv'
OUTPUT_FILE = 'steam_reviews_100k_balanced.csv'

TRAIN_SIZE = 80000
VAL_SIZE = 10000
TEST_SIZE = 10000
TOTAL_SIZE = TRAIN_SIZE + VAL_SIZE + TEST_SIZE

MIN_REVIEW_LENGTH = 20
MAX_REVIEW_LENGTH = 2000
MIN_VOTES_FOR_LABEL = 3
HELPFUL_THRESHOLD = 2

CHUNK_SIZE = 100000


def is_english_text(text):
    # Check if text is English using character analysis
    if pd.isna(text) or len(str(text).strip()) < 10:
        return False
    
    text = str(text)
    total_alpha = sum(1 for c in text if c.isalpha())
    if total_alpha < 5:
        return False
    
    # Reject non-English scripts (Cyrillic, CJK, Arabic, Thai, Greek, Hebrew)
    cyrillic = len(re.findall(r'[\u0400-\u04FF]', text))
    cjk = len(re.findall(r'[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF]', text))
    arabic = len(re.findall(r'[\u0600-\u06FF]', text))
    thai = len(re.findall(r'[\u0E00-\u0E7F]', text))
    greek = len(re.findall(r'[\u0370-\u03FF]', text))
    hebrew = len(re.findall(r'[\u0590-\u05FF]', text))
    
    if cyrillic + cjk + arabic + thai + greek + hebrew > 0:
        return False
    
    # Allow up to 2 accented characters (for borrowed words like café)
    european_chars = len(re.findall(
        r'[àáâãäåæçèéêëìíîïðñòóôõöøùúûüýÿßœąćęłńśźżřžščďťňěůĺľŕ]', 
        text, re.IGNORECASE
    ))
    if european_chars > 2:
        return False
    
    return True


def filter_non_english(df):
    # Filter dataframe to only include English reviews
    mask = df['review'].apply(is_english_text)
    filtered = df[mask]
    
    return filtered


def load_and_filter_data():
    # Load data in chunks and apply filters
    
    filtered_chunks = []
    total_processed = 0
    target_per_class = (TOTAL_SIZE // 2) * 2
    
    for i, chunk in enumerate(pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, low_memory=False)):
        total_processed += len(chunk)
        
        # Basic filters
        filtered = chunk[
            (chunk['language'] == 'english') &
            (chunk['review'].notna()) &
            (chunk['review'].str.len() >= MIN_REVIEW_LENGTH) &
            (chunk['review'].str.len() <= MAX_REVIEW_LENGTH) &
            ((chunk['votes_up'].fillna(0) + chunk['votes_funny'].fillna(0)) >= MIN_VOTES_FOR_LABEL) &
            (chunk['author_playtime_at_review'].notna()) &
            (chunk['voted_up'].notna()) &
            (chunk['timestamp_created'].notna()) &
            (chunk['received_for_free'].notna())
        ].copy()
        
        if len(filtered) > 0:
            filtered['label_helpful'] = (filtered['votes_up'] >= HELPFUL_THRESHOLD).astype(int)
            filtered_chunks.append(filtered)
        
        # Progress check
        if len(filtered_chunks) > 0:
            temp_df = pd.concat(filtered_chunks, ignore_index=True)
            n_helpful = (temp_df['label_helpful'] == 1).sum()
            n_not_helpful = (temp_df['label_helpful'] == 0).sum()
            
            print(f"  Chunk {i+1}: {len(temp_df):,} total ({n_helpful:,} helpful, {n_not_helpful:,} not helpful)")
            
            if n_helpful >= target_per_class and n_not_helpful >= target_per_class:
                break
    
    df = pd.concat(filtered_chunks, ignore_index=True)
    return df


def create_temporal_splits(df, train_size, val_size, test_size):
    # Create temporally-ordered splits with 50/50 balance in each
    # Both classes sampled from same time periods (prevents temporal leakage)
    
    df_sorted = df.sort_values('timestamp_created').reset_index(drop=True)
    
    train_per_class = train_size // 2
    val_per_class = val_size // 2
    test_per_class = test_size // 2
    
    # Get minority class (not-helpful) to determine time boundaries
    minority_df = df_sorted[df_sorted['label_helpful'] == 0].copy()
    total_minority_needed = train_per_class + val_per_class + test_per_class
    
    if len(minority_df) < total_minority_needed:
        scale = len(minority_df) / total_minority_needed
        train_per_class = int(train_per_class * scale * 0.95)
        val_per_class = int(val_per_class * scale * 0.95)
        test_per_class = int(test_per_class * scale * 0.95)
    
    # Time boundaries from minority class
    minority_train_end_time = minority_df.iloc[train_per_class - 1]['timestamp_created']
    minority_val_end_time = minority_df.iloc[train_per_class + val_per_class - 1]['timestamp_created']
    
    # Split by time boundaries
    train_period = df_sorted[df_sorted['timestamp_created'] <= minority_train_end_time]
    val_period = df_sorted[
        (df_sorted['timestamp_created'] > minority_train_end_time) & 
        (df_sorted['timestamp_created'] <= minority_val_end_time)
    ]
    test_period = df_sorted[df_sorted['timestamp_created'] > minority_val_end_time]
    
    def balanced_sample(period_df, n_per_class, seed):
        # Sample equal numbers of each class from a time period
        helpful = period_df[period_df['label_helpful'] == 1]
        not_helpful = period_df[period_df['label_helpful'] == 0]
        n_sample = min(n_per_class, len(helpful), len(not_helpful))
        
        helpful_sample = helpful.sample(n=n_sample, random_state=seed)
        not_helpful_sample = not_helpful.sample(n=n_sample, random_state=seed)
        combined = pd.concat([helpful_sample, not_helpful_sample])
        return combined.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    train_df = balanced_sample(train_period, train_per_class, seed=42)
    val_df = balanced_sample(val_period, val_per_class, seed=43)
    test_df = balanced_sample(test_period, test_per_class, seed=44)
    
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    return train_df, val_df, test_df

def main():
    # Load and filter
    df = load_and_filter_data()
    
    # Language filtering
    df = filter_non_english(df)
    
    # Create splits
    train_df, val_df, test_df = create_temporal_splits(
        df, TRAIN_SIZE, VAL_SIZE, TEST_SIZE
    )
    
    # Combine and save
    df_final = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    columns_to_keep = [
        'review', 'label_helpful', 'split', 'timestamp_created',
        'author_playtime_at_review', 'author_playtime_forever',
        'author_num_games_owned', 'author_num_reviews',
        'voted_up', 'received_for_free', 'steam_purchase',
        'written_during_early_access', 'app_name', 'app_id',
        'votes_up', 'votes_funny'
    ]
    columns_to_keep = [c for c in columns_to_keep if c in df_final.columns]
    df_final = df_final[columns_to_keep]
    
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df_final):,} reviews to {OUTPUT_FILE}")
    
    # Print some stats
    print("\nSizes:")
    for split in ['train', 'val', 'test']:
        split_df = df_final[df_final['split'] == split]
        helpful_pct = (split_df['label_helpful'] == 1).mean() * 100
        print(f"  {split}: {len(split_df):,} samples, {helpful_pct:.1f}% helpful")


if __name__ == '__main__':
    main()