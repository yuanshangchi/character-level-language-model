import torch

from nanoGPT_model import GPT, GPTConfig

# Load checkpoint
checkpoint_path = "src/ckpt.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# Build the model
model_args = checkpoint["model_args"]
model = GPT(GPTConfig(**model_args))
# model.load_state_dict(checkpoint["model"])
state_dict = checkpoint["model"]
if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
model.eval()

# Choose device (Apple Silicon support)
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

# Load vocab
if "config" in checkpoint and "stoi" in checkpoint["config"]:
    stoi = checkpoint["config"]["stoi"]
    itos = checkpoint["config"]["itos"]
else:
    # Fallback: rebuild from model's vocab_size
    vocab = list(
        " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\n"
    )
    vocab = list(" !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    assert len(vocab) <= model.config.vocab_size, "vocab too small for model size"
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
encode = lambda s: [stoi.get(c, stoi[" "]) for c in s]
decode = lambda l: "".join([itos.get(i, " ") for i in l])


# Inference functions
@torch.no_grad()
def generate_characters(prompt, num_characters=100, temperature=0.8, top_k=40):
    input_ids = torch.tensor([encode(prompt)], dtype=torch.long).to(device)

    for _ in range(num_characters):
        input_cond = (
            input_ids
            if input_ids.size(1) <= model.config.block_size
            else input_ids[:, -model.config.block_size :]
        )
        logits, _ = model(input_cond)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("Inf")

        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat((input_ids, next_id), dim=1)

    output_text = decode(input_ids[0].tolist())
    return output_text[len(prompt) :]  # Only new characters


def generate_next_character(prompt):
    return generate_characters(prompt, num_characters=1)


# Example usage
if __name__ == "__main__":
    prompt = "ROMEO:"
    next_char = generate_next_character(prompt)
    print(f"Next character: {next_char}")
    print(f"Full text: {prompt + next_char}")

    # generated = generate_characters(prompt, num_characters=200)
    # print("\nGenerated:")
    # print(prompt + generated)
