import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Set local directory
local_model_path = "./local_gpt2"

# Use MPS if on Mac with Apple Silicon
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load from local path — no internet needed
tokenizer = GPT2Tokenizer.from_pretrained(local_model_path)
LOCAL_GPT2_MODEL = GPT2LMHeadModel.from_pretrained(local_model_path).to(device)
LOCAL_GPT2_MODEL.eval()


def generate_characters(prompt, num_characters=100):
    """
    Generate a specified number of characters based on the input prompt.

    Args:
        prompt (str): The initial text to start the generation.
        num_characters (int): The number of characters to generate.

    Returns:
        str: The newly generated characters.
    """
    text = prompt
    for _ in range(num_characters):
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = LOCAL_GPT2_MODEL.generate(
                **inputs, max_new_tokens=1, do_sample=True, temperature=0.8, top_k=50
            )
        # Get the full decoded text and add just the next new character
        new_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        new_char = new_text[len(text) :][:1]  # Only the *next* character
        text += new_char

    # The newly generated characters
    return text[len(prompt) :]


def generate_next_character(prompt):
    """
    Generate the next character based on the input prompt.

    Args:
        prompt (str): The initial text to start the generation.

    Returns:
        str: The generated text.
    """
    return generate_characters(prompt, num_characters=1)


def generate_tokens(prompt, num_tokens=100):
    """
    Generate a specified number of tokens based on the input prompt.

    Args:
        prompt (str): The initial text to start the generation.
        num_tokens (int): The number of tokens to generate.

    Returns:
        str: The newly generated tokens.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = LOCAL_GPT2_MODEL.generate(
            **inputs,
            max_new_tokens=num_tokens,
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
        )
    # Get the full decoded text
    new_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # The newly generated tokens
    return new_text[len(prompt) :].strip()  # Remove the prompt part


# Example usage
if __name__ == "__main__":
    prompt = "Happy new ye"
    next_char = generate_next_character(prompt)
    print(f"Next character: {next_char}")
    print(f"Full text: {prompt + next_char}")

    next_tokens = generate_tokens(prompt, num_tokens=10)
    print(f"Next tokens: {next_tokens}")
    print(f"Full text: {prompt + next_tokens}")
