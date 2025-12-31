from tiktoken import get_encoding
import torch

from src.gpt import GptConfig, GptModel, generate_text_simple


# Encode sample input
tokenizer = get_encoding(encoding_name="gpt2")
start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print(f"{encoded=}")
encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dim
print(f"{encoded_tensor.shape=}")

# Create a small model for demo purposes
model_config = GptConfig(
    vocab_size=tokenizer.max_token_value + 1,
    context_length=10,
    emb_dim=768,
    n_heads=2,
    n_layers=2,
    drop_rate=0.1,
    qkv_bias=False,
)
model = GptModel(config=model_config)
model.eval()

# Generate new tokens
out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=model_config.context_length,
)
print(f"{out=}")
print("Output length:", len(out[0]))

# Decode the tokens
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)
