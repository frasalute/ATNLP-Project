import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        self.emb_dim = emb_dim  # if each token in the input is represented by a vector of 512 dimensions, emb_dim = 512.
        self.num_heads = num_heads # Multi-head attention splits the embedding into num_heads smaller parts, allowing the model to focus on different aspects of the input sequence
        self.head_dim = emb_dim // num_heads # dimension of each attention head

        # These layers project the input embeddings into separate query (Q), key (K), and value (V) spaces for each head.
        self.query_proj = nn.Linear(emb_dim, num_heads * self.head_dim)
        self.key_proj = nn.Linear(emb_dim, num_heads * self.head_dim)
        self.value_proj = nn.Linear(emb_dim, num_heads * self.head_dim)
        self.out_proj = nn.Linear(num_heads * self.head_dim, emb_dim) # After computing attention for each head, their outputs are concatenated.
        # The out_proj layer combines the outputs back into the original embedding size emb_dim.

    # To split Q - K - V in heads
    def _split_heads(self, hidden_states):
        batch_size, seq_len, emb_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )

        return hidden_states.permute(0, 2, 1, 3)

    # where we concatened all the results of our heads
    def _merge_heads(self, hidden_states):
        batch_size, num_heads, seq_len, head_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(
            batch_size, seq_len, self.num_heads * self.head_dim
        )

        return hidden_states

    # Performing the multi-head self attention calculations
    def forward(self, query, key, value, mask=None):
        # defining the vectors
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        # divide Q - K - V in heads
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        # dot product Q x K
        key_out = query @ key.transpose(-2, -1)

        # for the Decoder, apply a mask to not see the next words 
        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, -1e9)

        # apply softmax
        key_out = torch.softmax(key_out / self.head_dim**0.5, dim=-1)

        # dot product Key output x V
        attn = key_out @ value

        # Concatenate the heads together
        attn = self._merge_heads(attn)

        # concatenated all the single self-attention outputs back to the original embedding dimension
        attn_output = self.out_proj(attn)

        return attn_output

# Implements the core logic of the Transformer, including multi-head attention and feedforward processing, with normalization and skip connections
class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout, forward_dim):
        super().__init__()

        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.forward_dim = forward_dim

        self.norm = nn.LayerNorm(emb_dim, eps=1e-6)
        self.forward_norm = nn.LayerNorm(emb_dim, eps=1e-6)

        self.FNN = nn.Sequential(
            nn.Linear(self.emb_dim, self.forward_dim), # First Linear Layer: Expands the input embedding size (emb_dim) to a larger intermediate size (forward_dim)
            nn.ReLU(),                                 # Apply ReLU
            nn.Linear(self.forward_dim, self.emb_dim), # Second Linear Layer: Reduces the dimensionality back to the original embedding size, making it compatible with the rest of the Transformer architecture.
        )
        self.attn = MultiHeadAttention(self.emb_dim, self.num_heads)

    def forward(self, query, key, value, mask):
        # Attention
        attn = self.attn(query, key, value, mask)
        # Add & Norm
        attn = attn + query  # Skip con
        attn = self.dropout(attn)
        attn = self.norm(attn)

        # Feed Forward
        # While attention captures context across tokens, the feedforward network improves the representation of individual tokens by applying learned transformations.
        output = self.FNN(attn) # Nonlinear transformations
        # Add & Norm
        output = output + attn  # Skip con
        output = self.dropout(output)
        output = self.forward_norm(output)

        return output

# Positional Encoding
def get_sinusoid_table(max_len, emb_dim):
    def get_angle(pos, i, emb_dim):
        return pos / 10000 ** ((2 * (i // 2)) / emb_dim)

    sinusoid_table = torch.zeros(max_len, emb_dim)
    for pos in range(max_len):
        for i in range(emb_dim):
            if i % 2 == 0:
                sinusoid_table[pos, i] = math.sin(get_angle(pos, i, emb_dim))
            else:
                sinusoid_table[pos, i] = math.cos(get_angle(pos, i, emb_dim))
    return sinusoid_table

# Combines embeddings, positional encodings, and multiple TransformerBlock layers
class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim,
        num_layers,
        num_heads,
        forward_dim,
        dropout,
        max_len,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.forward_dim = forward_dim
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len

        # Positional Encoding 
        pos_weight = get_sinusoid_table(max_len + 1, emb_dim)
        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding.from_pretrained(pos_weight, freeze=True)

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(emb_dim, num_heads, dropout, forward_dim)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask):
        seq_len = x.size(1)
        tok_emb = self.tok_emb(x).to(device=x.device)
        pos_indices = torch.arange(1, seq_len + 1, device=x.device)  # [seq_len]
        pos_emb = self.pos_emb(pos_indices)
        embedding = tok_emb + pos_emb.unsqueeze(0)
        embedding = self.dropout(embedding)

        for block in self.transformer_blocks:
            output = block(embedding, embedding, embedding, mask)

        return output

# Masked Multi-Head Self-Attention -> Cross-Attention with Encoder Output -> Feedforward Network
class DecoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, forward_dim, dropout):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.forward_dim = forward_dim
        self.dropout = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(emb_dim, eps=1e-6)
        self.attn = MultiHeadAttention(emb_dim, num_heads)
        self.transformer_block = TransformerBlock(
            emb_dim, num_heads, dropout, forward_dim
        )

    def forward(self, x, value, key, src_mask, tgt_mask):
        # Masked Self-Attention 
        attn = self.attn(x, x, x, tgt_mask)
        # Add & Norm
        attn = attn + x
        attn = self.dropout(attn)
        attn = self.norm(attn)

        # Cross-Attention with Encoder Output - V and K come from the encoder output
        output = self.transformer_block(attn, value, key, src_mask)

        return output


class Decoder(nn.Module):
    def __init__(
        self, vocab_size, emb_dim, num_layers, num_heads, forward_dim, dropout, max_len
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.forward_dim = forward_dim
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len

        pos_weight = get_sinusoid_table(max_len + 1, emb_dim)
        self.pos_emb = nn.Embedding.from_pretrained(pos_weight, freeze=True)
        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(emb_dim, num_heads, forward_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.out_layer = nn.Linear(emb_dim, vocab_size)


    def forward(self, x, encoder_out, src_mask, tgt_mask):
        # input x is the taget embedding
        tok_emb = self.tok_emb(x)
        seq_len = x.size(1)
        pos_indices = torch.arange(1, seq_len + 1, device=x.device)  
        pos_emb = self.pos_emb(pos_indices)  

        embedding = tok_emb + pos_emb.unsqueeze(0)  
        embedding = self.dropout(embedding)

        # Pass Through Decoder Blocks
        for block in self.decoder_blocks:
            output = block(embedding, encoder_out, encoder_out, src_mask, tgt_mask)

        output = self.out_layer(embedding) 
        return output


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        src_pad_idx,
        tgt_pad_idx,
        emb_dim=512,
        num_layers=6,
        num_heads=8,
        forward_dim=2048,
        dropout=0.0,
        max_len=128,
    ):
        super().__init__()

        self.encoder = Encoder(
            src_vocab_size,
            emb_dim,
            num_layers,
            num_heads,
            forward_dim,
            dropout,
            max_len,
        )
        self.decoder = Decoder(
            tgt_vocab_size,
            emb_dim,
            num_layers,
            num_heads,
            forward_dim,
            dropout,
            max_len,
        )

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

    # input mask
    # Ensures that padding tokens in the source sequence are ignored during attention
    def create_src_mask(self, src):
        device = src.device
        # (batch_size, 1, 1, src_seq_len)
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(device)
    # output mask
    # Ensures that padding tokens in the source sequence are ignored during attention
    # Ensures that tokens can only attend to earlier tokens
    def create_tgt_mask(self, tgt):
        device = tgt.device
        batch_size, tgt_len = tgt.shape
        tgt_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_mask = tgt_mask * torch.tril(torch.ones((tgt_len, tgt_len))).expand(
            batch_size, 1, tgt_len, tgt_len
        ).to(device)
        return tgt_mask

    # directing the sentence to the Ecoder and Decoder
    def forward(self, src, tgt):
        src_mask = self.create_src_mask(src)
        tgt_mask = self.create_tgt_mask(tgt)

        encode_out = self.encoder(src, src_mask)
        decode_out = self.decoder(tgt, encode_out, src_mask, tgt_mask)

        return decode_out

# Verifies the correctness of the Transformer
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Transformer(
        src_vocab_size=200,
        tgt_vocab_size=220,
        src_pad_idx=0,
        tgt_pad_idx=0,
    ).to(device)

    # source input: batch size 4, sequence length of 75
    src_in = torch.randint(0, 200, (4, 75)).to(device)

    # target input: batch size 4, sequence length of 80
    tgt_in = torch.randint(0, 220, (4, 80)).to(device)

    # expected output shape of the model
    expected_out_shape = torch.Size([4, 80, 220])

    with torch.no_grad():
        out = model(src_in, tgt_in)

    assert (
        out.shape == expected_out_shape
    ), f"wrong output shape, expected: {expected_out_shape}"

    print("Passed test!")


if __name__ == "__main__":
    main()