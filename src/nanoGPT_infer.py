import torch
from transformers import AutoTokenizer  # Import the tokenizer
from nanoGPT_model import GPT, GPTConfig

# Load the saved state dictionary
checkpoint_path = "src/gpt_multilingual_unicode.pt"  # Path to your checkpoint file
state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)  # Load the state dictionary

# Recreate the model with the same configuration
config = GPTConfig(vocab_size=149021, block_size=512, n_layer=8,
    n_head=8,                  # heads
    n_embd=512,                # hidden size
    dropout=0.1)
model = GPT(config)

# Load the state dictionary into the model
model.load_state_dict(state_dict.get("model_state_dict", state_dict), strict=False)

# Choose device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

model = model.to(device)
model.eval()  # Set the model to evaluation mode
# Initialize the tokenizer (from practice.py)
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

# print(f"Built printable Unicode vocabulary with {vocab_size} characters.")

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

tokenizer = PrintableUnicodeTokenizer(char_vocab)



# Inference functions
@torch.no_grad()
def generate_characters(prompt, num_characters=100, temperature=0.8, top_k=40):
    # Encode the prompt using UnicodeTokenizer
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    for _ in range(num_characters):
        # Truncate input to the model's block size
        input_cond = (
            input_ids
            if input_ids.size(1) <= model.config.block_size
            else input_ids[:, -model.config.block_size :]
        )

        # Forward pass
        logits = model(input_cond)
        if isinstance(logits, tuple):  # Handle tuple outputs
            logits = logits[0]
        logits = logits[:, -1, :] / temperature  # Focus on the last token

        # Apply top-k sampling
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("Inf")

        # Convert logits to probabilities and sample the next token
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        # Append the sampled token to the input
        input_ids = torch.cat((input_ids, next_id), dim=1)

    # Decode the generated sequence using UnicodeTokenizer
    output_text = tokenizer.decode(input_ids[0].tolist())
    # print(f"Generated sequence: {output_text}")
    return output_text[len(prompt) :]  # Return only the newly generated characters


def generate_next_character(prompt):
    return generate_characters(prompt, num_characters=1)


# Example usage
if __name__ == "__main__":
    prompt = "ROMEO:"
    next_char = generate_next_character(prompt)
    print(f"Next character: {next_char}")
    print(f"Full text: {prompt + next_char}")

    generated = generate_characters(prompt, num_characters=200)
    print("\nGenerated:")
    print(prompt + generated)
