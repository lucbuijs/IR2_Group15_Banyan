# Script to subsample training data to ~10M BPEmb tokens
# Run with: python scripts/subsample_data.py

from bpemb import BPEmb
from tqdm import tqdm
import random

TARGET_TOKENS = 10_000_000
random.seed(42)  # For reproducibility

bpe = BPEmb(lang='en', vs=25000, dim=100)

print("Loading training data...")
with open('../data/en_train.txt', 'r') as f:
    lines = f.readlines()

print(f"Original lines: {len(lines):,}")

# Shuffle and accumulate until we hit target
random.shuffle(lines)
selected_lines = []
total_tokens = 0

print(f"Selecting lines to reach ~{TARGET_TOKENS:,} tokens...")
for line in tqdm(lines):
    tokens = bpe.encode_ids(line.strip())
    if total_tokens + len(tokens) > TARGET_TOKENS:
        break
    selected_lines.append(line)
    total_tokens += len(tokens)

print(f"\n{'='*50}")
print(f"Selected lines: {len(selected_lines):,}")
print(f"Total tokens: {total_tokens:,}")
print(f"{'='*50}")

# Write subsampled data
output_path = '../data/en_train_10m.txt'
with open(output_path, 'w') as f:
    f.writelines(selected_lines)

print(f"\n✅ Saved subsampled data to: {output_path}")
print(f"Update your train.job to use --train_path data/en_train_10m.txt")
