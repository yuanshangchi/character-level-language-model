import random
import time
import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
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


# 3. Tokenizer setup (Unicode Encoding)
class UnicodeTokenizer:
    def __init__(self):
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = 1

    def encode(self, text, max_length=None, padding=False, truncation=False):
        # Convert each character to its Unicode code point
        tokens = [ord(char) for char in text]
        if truncation and max_length:
            tokens = tokens[:max_length]
        if padding and max_length:
            tokens += [self.pad_token_id] * (max_length - len(tokens))
        tokens.append(self.eos_token_id)  # Add EOS token
        return tokens

    def decode(self, token_ids):
        # Convert Unicode code points back to characters
        return "".join(chr(token) for token in token_ids if token > 1)  # Ignore pad and eos tokens

tokenizer = UnicodeTokenizer()

# Update the model configuration for Unicode encoding
vocab_size = 1114112  # Maximum Unicode code point (0x10FFFF) + 2 for pad and eos tokens
block_size = 128  # Set the block size explicitly
config = GPTConfig(vocab_size=vocab_size, block_size=block_size)

# 4. Sampler setup
languages = ["en", "es", "hi"]
lang_probs = [0.5, 0.3, 0.2]

stream_sampler = MultilingualStreamProportionalSampler(languages, lang_probs)


# 5. Collate function
def collate(batch):
    # Tokenize the text field in the batch using Unicode encoding
    tokenized_batch = [
        tokenizer.encode(item["text"], truncation=True, padding=True, max_length=block_size - 2) + [tokenizer.eos_token_id]
        for item in batch
    ]

    # Stack the input_ids into a tensor
    input_ids = torch.tensor(tokenized_batch, dtype=torch.long)
    targets = input_ids.clone()
    return input_ids, targets

train_loader = DataLoader(stream_sampler, batch_size=2, collate_fn=collate)

# 6. Instantiate model and optimizer
device = "cuda" if torch.cuda.is_available() else "cpu"

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
        # print(f"Outputs structure at step {step}: {type(outputs)}, {outputs}")  # Debugging

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
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error during training at step {step}: {e}")
        traceback.print_exc()
        continue


# Save the trained model
torch.save(model, "gpt_multilingual_unicode.pt")
print("Model saved to gpt_multilingual_unicode.pt")
