import torch
from transformers import AutoTokenizer  # Import the tokenizer
from nanoGPT_model import GPT, GPTConfig

# Load the saved state dictionary
checkpoint_path = "src/gpt_multilingual_unicode.pt"  # Path to your checkpoint file
state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)  # Load the state dictionary

# Recreate the model with the same configuration
config = GPTConfig(vocab_size=1114112, block_size=128)
model = GPT(config)

# Load the state dictionary into the model
model.load_state_dict(state_dict.state_dict())

# Choose device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

model = model.to(device)
model.eval()  # Set the model to evaluation mode
# Initialize the tokenizer (from practice.py)
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



# Inference functions
@torch.no_grad()
def generate_characters(prompt, num_characters=100, temperature=0.8, top_k=40):
    # Encode the prompt using UnicodeTokenizer
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    print(f"Encoded input: {input_ids}")

    for _ in range(num_characters):
        # Truncate input to the model's block size
        input_cond = (
            input_ids
            if input_ids.size(1) <= model.config.block_size
            else input_ids[:, -model.config.block_size :]
        )
        print(f"Input condition shape: {input_cond.shape}")

        # Forward pass
        logits = model(input_cond)
        if isinstance(logits, tuple):  # Handle tuple outputs
            logits = logits[0]
        logits = logits[:, -1, :] / temperature  # Focus on the last token
        print(f"Logits (last token): {logits}")

        # Apply top-k sampling
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("Inf")

        # Convert logits to probabilities and sample the next token
        probs = torch.nn.functional.softmax(logits, dim=-1)
        print(f"Probabilities: {probs}")
        # Check if all probabilities are 0
        for prob in probs:
            count = 0
            for p in prob:
                if p != 0:
                    print("Probability is {p} {count}", p , count)
                count += 1
        next_id = torch.multinomial(probs, num_samples=1)
        print(f"Sampled token ID: {next_id}")

        # Append the sampled token to the input
        input_ids = torch.cat((input_ids, next_id), dim=1)

    # Decode the generated sequence using UnicodeTokenizer
    print(f"Generated input IDs: {input_ids}")
    output_text = tokenizer.decode([82, 79, 77, 69, 79, 58,  1,])
    print(f"Generated sequence: {output_text}")
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
