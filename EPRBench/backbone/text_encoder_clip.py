import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

class CLIPTextEncoder(nn.Module):
    """
    Text encoder using OpenCLIP:
    - forward(texts): Sentence embedding (B, output_dim)
    - encode_batch_tokens(texts): Token-level embeddings (B, L, output_dim) and attention mask (B, L)
    """
    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str = "laion2b_s34b_b88k",
        output_dim: int = 768,
        freeze_clip: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.output_dim = output_dim

        # Load CLIP model and tokenizer
        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # CLIP text projection matrix: width -> clip_embed_dim
        # encode_text output dimension is usually clip_embed_dim (e.g. 512/768/1024)
        clip_embed_dim = self.model.text_projection.shape[1]

        # Project sentence embedding to target dimension
        self.text_proj = nn.Sequential(
            nn.Linear(clip_embed_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim, eps=1e-6),
        )

        if freeze_clip:
            for p in self.model.parameters():
                p.requires_grad = False

    def _tokenize(self, texts):
        tokens = self.tokenizer(texts)
        return tokens

    def forward(self, texts):
        """
        Return sentence embedding (B, output_dim), L2 normalized.
        """
        device = next(self.parameters()).device
        tokens = self._tokenize(texts).to(device)

        # CLIP sentence embedding
        with torch.no_grad() if not any(p.requires_grad for p in self.model.parameters()) else torch.enable_grad():
            clip_sent = self.model.encode_text(tokens)  # (B, clip_embed_dim)

        sent = self.text_proj(clip_sent)               # (B, output_dim)
        sent = F.normalize(sent, p=2, dim=1)
        return sent

    def encode_batch_tokens(self, texts):
        """
        Return token-level embeddings and mask:
        - token_embeddings: (B, L, output_dim)
        - attention_mask:   (B, L) 1 for valid, 0 for padding
        """
        device = next(self.parameters()).device
        tokens = self._tokenize(texts).to(device)           # (B, L)
        # OpenCLIP padding id is 0
        attention_mask = (tokens != 0).long()               # (B, L)

        # Get hidden states for each token
        with torch.no_grad() if not any(p.requires_grad for p in self.model.parameters()) else torch.enable_grad():
            x = self.model.token_embedding(tokens)          # (B, L, width)
            x = x + self.model.positional_embedding.to(x.dtype)  # (L, width) -> broadcast to (B, L, width)
            x = x.permute(1, 0, 2)                          # (L, B, width)
            x = self.model.transformer(x)                   # (L, B, width)
            x = x.permute(1, 0, 2)                          # (B, L, width)
            x = self.model.ln_final(x)                      # (B, L, width)

            x = x @ self.model.text_projection              # (B, L, clip_embed_dim)

        token_embeddings = self.text_proj(x)                # (B, L, output_dim)
        token_embeddings = F.normalize(token_embeddings, p=2, dim=-1)
        return token_embeddings, attention_mask

    def encode_text_and_tokens(self, texts):
        """
        Return both sentence embedding and token-level embeddings/mask
        """
        sent = self.forward(texts)
        tok, mask = self.encode_batch_tokens(texts)
        return sent, tok, mask


def get_clip_encoder(
    output_dim: int = 768,
    device: str = "cuda",
    freeze: bool = True,
    model_name: str = "ViT-B-16",
    pretrained: str = "laion2b_s34b_b88k",
):
    enc = CLIPTextEncoder(model_name=model_name, pretrained=pretrained, output_dim=output_dim, freeze_clip=freeze)
    return enc.to(device)
