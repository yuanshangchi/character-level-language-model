import random
import time
import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from nanoGPT_model import GPT, GPTConfig  # assumes you have model.py with GPT classes defined
import traceback

# 1. Load multilingual C4 using streaming with retry logic and User-Agent
def load_with_retries():
    retries = 5
    for i in range(retries):
        try:
            return load_dataset(
                "allenai/c4",
                name="multilingual",
                streaming=True,
                split="train",
            )
        except Exception as e:
            print(f"Retry {i + 1}/{retries} failed: {e}")
            time.sleep(2 ** i)  # Exponential backoff
    raise RuntimeError("Failed to load dataset after retries")

dataset = load_with_retries()

# 2. Define MultilingualStreamProportionalSampler
class MultilingualStreamProportionalSampler(IterableDataset):
    def __init__(self, languages, lang_probs):
        self.datasets = {
            lang: load_dataset(
                "allenai/c4",
                name=f"{lang}",
                split="train",
                streaming=True,
            )
            for lang in languages
        }
        self.lang_probs = lang_probs
        self.languages = languages

    def __iter__(self):
        while True:
            try:
                lang = random.choices(self.languages, weights=self.lang_probs)[0]
                data = next(iter(self.datasets[lang]))
                yield data
            except StopIteration:
                continue  # Handle streaming dataset disconnections gracefully

# 3. Tokenizer setup
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# 4. Sampler setup
languages = ["en", "es", "fr", "de", "zh"]
lang_probs = [0.5, 0.2, 0.1, 0.1, 0.1]

stream_sampler = MultilingualStreamProportionalSampler(languages, lang_probs)

# 5. Collate function
def collate(batch):
    # Debugging: Print the structure of the batch
    # print("Batch structure:", batch)

    # Tokenize the text field in the batch
    tokenized_batch = [tokenizer(item["text"], truncation=True, padding="max_length", max_length=1024) for item in batch]
    # Debugging: Print the tokenized batch
    print("Tokenized batch:", tokenized_batch)
    # Check if tokenized_batch is empty
    if not tokenized_batch:
        print("Tokenized batch is empty")
        return None, None
    # Stack the input_ids into a tensor
    input_ids = torch.stack([torch.tensor(item["input_ids"]) for item in tokenized_batch])
    targets = input_ids.clone()
    return input_ids, targets

train_loader = DataLoader(stream_sampler, batch_size=4, collate_fn=collate)

# 6. Instantiate model and optimizer
device = "cuda" if torch.cuda.is_available() else "cpu"

config = GPTConfig()
model = GPT(config).to(device)

optimizer = model.configure_optimizers(
    weight_decay=0.1, learning_rate=3e-4, betas=(0.9, 0.95), device_type=device
)

# 7. Training loop
model.train()
for step, (input_ids, targets) in enumerate(train_loader):
    try:
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(input_ids, targets)
        print(f"Outputs structure at step {step}: {type(outputs)}, {outputs}")  # Debugging

        # Handle outputs based on their structure
        if isinstance(outputs, torch.Tensor):
            loss = outputs  # If the model directly returns the loss as a tensor
        elif isinstance(outputs, tuple):
            loss = outputs[1]  # Assuming loss is the second element of the tuple
        elif isinstance(outputs, dict):
            loss = outputs["loss"]  # If the model returns a dictionary with a "loss" key
        else:
            raise ValueError(f"Unexpected outputs structure: {type(outputs)}")

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss.item()}")

        # Stop after 1000 steps
        if step == 1000:
            break
    except Exception as e:
        print(f"Error during training at step {step}: {e}")
        traceback.print_exc()
        continue

# Save the trained model
torch.save(model.state_dict(), "gpt_multilingual.pt")
print("Model saved to gpt_multilingual.pt")
