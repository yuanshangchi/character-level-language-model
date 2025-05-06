# Installation and Usage of OSCAR Dataset

## Installation

```bash
pip install datasets
```

## Usage

```python
from datasets import load_dataset

dataset = load_dataset(
    "oscar",
    "unshuffled_deduplicated_en",
    split="train",
    streaming=True,
    cache_dir="cache", # optional, specify a cache directory
    trust_remote_code=True, # very important to do streaming, otherwise it will download the entire dataset
)

sample_dataset = []
for i, example in enumerate(dataset):
    if i >= 10000:
        break
    sample_dataset.append(example)

# Print the first example to verify
print(sample_dataset[0])
```
