from pydantic import BaseModel, Field


class GptConfig(BaseModel):
    vocab_size: int = Field(
        default=50257,
        description="The number of tokens in the vocabulary",
    )
    context_length: int = Field(
        default=1024,
        description="The maximum number of input tokens the model can handle",
    )
    emb_dim: int = Field(
        default=768,
        description="The embedding size for each token",
    )
    n_heads: int = Field(
        default=12,
        description="Number of heads in the multi-head attention mechanism",
    )
    n_layers: int = Field(
        default=12,
        description="Number of transformer blocks in the model",
    )
    drop_rate: float = Field(
        default=0.1,
        description="Percent of hidden units to drop in each layer",
    )
    qkv_bias: bool = Field(
        default=False,
        description="Whether to include a bias vector in the linear layers",
    )
