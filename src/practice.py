import os
import pickle
import random
import time
import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from nanoGPT_model import GPT, GPTConfig
import traceback

# 1. Load multilingual C4 using streaming with retry logic

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
            print(f"Retry {i + 2}/{retries} failed: {e}")
            time.sleep(2 ** (i+4))
    raise RuntimeError("Failed to load dataset after retries")

dataset = load_with_retries()

# 2. Define MultilingualStreamProportionalSampler with Caching

class MultilingualStreamProportionalSampler(IterableDataset):
    def __init__(self, languages, lang_probs, cache_dir="cache"):
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
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self, lang):
        return os.path.join(self.cache_dir, f"{lang}_cache.pkl")

    def _load_from_cache(self, lang):
        cache_path = self._cache_path(lang)
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        return None

    def _save_to_cache(self, lang, data):
        cache_path = self._cache_path(lang)
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

    def __iter__(self):
        while True:
            try:
                lang = random.choices(self.languages, weights=self.lang_probs)[0]
                cached_data = self._load_from_cache(lang)
                if cached_data:
                    yield random.choice(cached_data)
                else:
                    data = []
                    for item in self.datasets[lang]:
                        data.append(item)
                        if len(data) >= 1000:
                            break
                    self._save_to_cache(lang, data)
                    yield random.choice(data)
            except StopIteration:
                continue

# 3. Tokenizer setup (Unicode Encoding)
import unicodedata

def is_printable(char):
    if char in ['\n', '\t', ' ']:
        return True
    category = unicodedata.category(char)
    return category[0] in ['L', 'N', 'P', 'S', 'Z', 'M']

def build_printable_vocab():
    # Add special tokens
    printable_chars = ['<pad>', '<eos>', '<unk>']

    # Optional: Restrict to Basic Multilingual Plane (0x0000–0xFFFF)
    for i in range(0x110000):  # You can switch to 0x110000 for full Unicode
        char = chr(i)
        if is_printable(char):
            printable_chars.append(char)
    return printable_chars

char_vocab = build_printable_vocab()
vocab_size = len(char_vocab)

print(f"Built printable Unicode vocabulary with {vocab_size} characters.")

class PrintableUnicodeTokenizer:
    def __init__(self, vocab, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>"):
        self.vocab = vocab
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for i, ch in enumerate(vocab)}
        self.pad_token_id = self.stoi[pad_token]
        self.eos_token_id = self.stoi[eos_token]
        self.unk_token_id = self.stoi[unk_token]

    def encode(self, text, max_length=None, padding=False, truncation=False):
        tokens = [self.stoi.get(ch, self.unk_token_id) for ch in text]

        if truncation and max_length:
            tokens = tokens[:max_length - 1]
            tokens.append(self.eos_token_id)
        elif max_length and padding:
            tokens.append(self.eos_token_id)

        if padding and max_length:
            tokens += [self.pad_token_id] * (max_length - len(tokens))

        return tokens

    def decode(self, token_ids):
        return ''.join(self.itos.get(i, '') for i in token_ids if i != self.pad_token_id)

# Create tokenizer instance
tokenizer = PrintableUnicodeTokenizer(char_vocab)
test_string = "Hello, 🌍!\n"
encoded = tokenizer.encode(test_string)
decoded = tokenizer.decode(encoded)

print("Original:", repr(test_string))
print("Encoded:", encoded)
print("Decoded:", repr(decoded))

# vocab_size = 1114112
block_size = 512
config = GPTConfig(
    vocab_size=vocab_size,
    block_size=block_size,            # increase context
    n_layer=8,                 # depth
    n_head=8,                  # heads
    n_embd=512,                # hidden size
    dropout=0.1                # regularization
)

# languages = ["en"] #, "es", "hi"]
# lang_probs = [1.0] #, 0.3, 0.2]

languages = [
    "en",  # English
    "zh",  # Chinese
    "es",  # Spanish
    "hi",  # Hindi
    "ar",  # Arabic
    "fr",  # French
    "pt",  # Portuguese
    "ru",  # Russian
    "ja",  # Japanese
    "de",  # German
    "bn",  # Bengali
    "ur",  # Urdu
    "tr",  # Turkish
    "vi",  # Vietnamese
    "sw",  # Swahili
    "ko",  # Korean
]

lang_probs = [
    0.22,  # en
    0.18,  # zh
    0.10,  # es
    0.07,  # hi
    0.06,  # ar
    0.05,  # fr
    0.05,  # pt
    0.05,  # ru
    0.04,  # ja
    0.04,  # de
    0.03,  # bn
    0.02,  # ur
    0.02,  # tr
    0.01,  # vi
    0.01,  # sw
    0.01,  # ko
]

# Normalize to ensure sum is 1.0
total = sum(lang_probs)
lang_probs = [p / total for p in lang_probs]



stream_sampler = MultilingualStreamProportionalSampler(languages, lang_probs, cache_dir="cache")

# 5. Collate function
import random
import torch

def collate(batch, block_size=128, max_retries=10):
    batch_inputs = []
    batch_targets = []

    for item in batch:
        tokens = tokenizer.encode(item["text"], truncation=False, padding=False)
        if len(tokens) < 2:
            continue  # skip too-short samples

        max_input_len = min(len(tokens) - 1, block_size)  # last token index is target

        # Find candidate input lengths where target token is not pad token and not space
        candidate_input_lens = [
            i for i in range(1, max_input_len + 1)
            if tokens[i] != tokenizer.pad_token_id and tokenizer.itos.get(tokens[i], '') != ' '
        ]

        if not candidate_input_lens:
            # fallback if no suitable target found: just pick a random length < max_input_len
            input_len = random.randint(1, max_input_len)
        else:
            input_len = random.choice(candidate_input_lens)

        input_chunk = tokens[:input_len]
        target_token = tokens[input_len]
        # print(f"Selected input length: {input_len}, target token: {target_token}")
        # print(f"Decoded input: {tokenizer.decode(input_chunk)}")
        # print(f"Decoded target: {tokenizer.decode([target_token])}")
        # Pad input chunk to block size
        input_padded = input_chunk + [tokenizer.pad_token_id] * (block_size - len(input_chunk))

        batch_inputs.append(input_padded)
        batch_targets.append(target_token)

    if len(batch_inputs) == 0:
        batch_inputs = [[tokenizer.pad_token_id] * block_size]
        batch_targets = [tokenizer.pad_token_id]

    input_ids = torch.tensor(batch_inputs, dtype=torch.long)
    target_ids = torch.tensor(batch_targets, dtype=torch.long).unsqueeze(1)

    return input_ids, target_ids




train_loader = DataLoader(stream_sampler, batch_size=512, collate_fn=collate)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT(config).to(device)

optimizer = model.configure_optimizers(
    weight_decay=0.01, learning_rate=1e-4, betas=(0.9, 0.95), device_type=device
)

criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, label_smoothing=0.1)

model.train()
avg_loss = 0.0
validation_losses = []
gradient_accumulation_steps = 4  # Adjust as needed for larger batches
total_steps = gradient_accumulation_steps * 1000  # Total steps to run

from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)

for step, (inputs, targets) in enumerate(train_loader):
    try:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        output = model(inputs)
        logits = output if isinstance(output, torch.Tensor) else output[0]

        B, T, V = logits.shape
        assert targets.shape == (B, T), f"Shape mismatch: logits ({B}, {T}, {V}), targets {targets.shape}"

        logits_flat = logits.view(-1, V)
        targets_flat = targets.view(-1)

        loss = criterion(logits_flat, targets_flat)
        loss = loss / gradient_accumulation_steps  # Scale loss for gradient accumulation
        avg_loss = 0.9 * avg_loss + 0.1 * loss.item()

        # optimizer.zero_grad()
        loss.backward()
        # optimizer.step()

        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        if step % 10*gradient_accumulation_steps == 0:
            print(f"Step {step}, Smoothed Loss: {avg_loss:.4f}")

        if step % 1000*gradient_accumulation_steps == 0 and step > 0:
            with torch.no_grad():
                model.eval()
                val_inputs, val_targets = next(iter(train_loader))
                val_inputs = val_inputs.to(device)
                val_targets = val_targets.to(device)
                val_targets[val_targets == tokenizer.eos_token_id] = tokenizer.pad_token_id

                val_output = model(val_inputs)
                val_logits = val_output if isinstance(val_output, torch.Tensor) else val_output[0]
                val_B, val_T, val_V = val_logits.shape
                assert val_targets.shape == (val_B, val_T), f"Validation shape mismatch: logits ({val_B}, {val_T}, {val_V}), targets {val_targets.shape}"

                val_logits_flat = val_logits.view(-1, val_V)
                val_targets_flat = val_targets.view(-1)
                val_loss = criterion(val_logits_flat, val_targets_flat).item()

                # Decode the predicted token
                predicted_token_ids = torch.argmax(val_logits, dim=-1)  # Get the most likely token IDs
                predicted_tokens = predicted_token_ids[:, -1].tolist()  # Take the last token in each sequence
                predicted_chars = [tokenizer.decode([token]) for token in predicted_tokens]  # Decode to characters

                # Decode the ground truth target
                target_tokens = val_targets[:, -1].tolist()  # Take the last token in each sequence
                target_chars = [tokenizer.decode([token]) for token in target_tokens]

                print(f"Validation at step {step}, Loss: {val_loss:.4f}")
                for i, (pred_char, target_char) in enumerate(zip(predicted_chars, target_chars)):
                    print(f"Sample {i}: Predicted: '{pred_char}', Target: '{target_char}'")

                validation_losses.append((step, val_loss))
                model.train()

        if step >= total_steps:
            break

    except Exception as e:
        print(f"Error during training at step {step}: {e}")
        traceback.print_exc()
        break


# Save the trained model and optimizer state
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "config": config,
}, "gpt_multilingual_unicode.pt")
print("Model and optimizer state saved to gpt_multilingual_unicode_checkpoint.pt")
