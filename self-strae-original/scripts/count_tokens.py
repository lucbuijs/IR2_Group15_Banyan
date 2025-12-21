# Script to count BPEmb tokens in the training data
# Run with: python scripts/count_tokens.py

from bpemb import BPEmb
from tqdm import tqdm

bpe = BPEmb(lang='en', vs=25000, dim=100)
total_tokens = 0

print("Loading training data...")
with open('../data/en_train_10m.txt', 'r') as f:
    lines = f.readlines()

print(f"Processing {len(lines):,} lines...")
for line in tqdm(lines):
    tokens = bpe.encode_ids(line.strip())
    total_tokens += len(tokens)

print(f'\n{"="*50}')
print(f'Total lines: {len(lines):,}')
print(f'Total BPEmb tokens: {total_tokens:,}')
print(f'Target (per BANYAN paper): ~10,000,000 tokens')
print(f'Difference: {total_tokens - 10_000_000:+,} tokens')
print(f'{"="*50}')

if 9_000_000 <= total_tokens <= 11_000_000:
    print("✅ Token count is within expected range!")
else:
    print(f"⚠️  Token count differs from expected ~10M")
