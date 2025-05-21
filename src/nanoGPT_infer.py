import torch
from transformers import AutoTokenizer  # Import the tokenizer
from nanoGPT_model import GPT, GPTConfig

# Load checkpoint
checkpoint_path = "src/gpt_multilingual.pt"  # Path to your checkpoint file
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# Build the model
model = GPT(GPTConfig())
model.load_state_dict(checkpoint)
model.eval()

# Choose device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)

# Initialize the tokenizer (from practice.py)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token



# Inference functions
@torch.no_grad()
def generate_characters(prompt, num_characters=100, temperature=0.8, top_k=40):
    # Encode the prompt
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

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

        print(f"Input IDs: {input_ids}")
        print(f"Logits shape: {logits.shape}")
        print(f"Top-K logits: {logits}")
        print(f"Sampled next ID: {next_id}")

    # Decode the generated sequence
    output_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
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
