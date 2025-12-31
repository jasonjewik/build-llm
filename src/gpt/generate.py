import torch

from src.gpt.model import GptModel


def generate_text_simple(
    model: GptModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
) -> torch.Tensor:
    """
    Generates tokens.

    :param model: a GPT model
    :param idx: a (batch, n_tokens) array of indices in the current context
    :param max_new_tokens: the max number of tokens to generate
    :param context_size: crops current context if it exceeds the supported
        context size (e.g., if the LLM supports only 5 tokens, and the current
        context size is 10, then only the last 5 tokens are used)

    :return: sequence of token indices of shape (batch, n_tokens)
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model.forward(idx_cond)
        # Focuses only on the last time step, so that (batch, n_token,
        # vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
