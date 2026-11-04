import torch
from torch import nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)
eval_iters = 20
n_embd = 192
n_head = 4
head_size = n_embd // n_head
n_layer = 4
dropout = 0.2
train_model = False  # Set to False after the first run saves the model
checkpoint_path = "model.pth"
# -------------

torch.manual_seed(1337)

# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# all unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers and vice versa
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


# encoder and decoder functions
def encode(text: str):
    return [stoi[x] for x in text]


def decode(code: list[int]):
    return "".join(itos[i] for i in code)


# train and data split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B,T,C) @ (B,C,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B,T,T)
        wei = F.softmax(wei, dim=-1)  # (B,T,T)
        wei = self.dropout(wei)
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # embedding layer of size vocab_size x n_embd
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # embedding layer of size block_size x n_embd
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) # stack of 2 transformer blocks
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size) # linear layer of size n_embd x vocab_size

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # get the embeddings for the input indices with dimensions (B, T, C) where B is batch size, T is sequence length, and C is embedding dimension
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # get the position embeddings for the input indices with dimensions (T, C)
        x = tok_emb + pos_emb # add the token embeddings and position embeddings together with dimensions
        x = self.blocks(x) # pass the embeddings through the transformer blocks with dimensions (B, T, C)
        x = self.ln_f(x) # apply layer normalization to the output of the transformer blocks with dimensions (B, T, C)
        logits = self.lm_head(x) # get the logits for the input embeddings with dimensions (B, T, vocab_size)
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  # crop context to the last block_size tokens
            logits, loss = self(idx_cond)  # get the predictions
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx



model = BigramLanguageModel()
m = model.to(device)

# create pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

if train_model:
    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
          losses = estimate_loss()
          print(
              f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
          )

        xb, yb = get_batch("train")

        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save(m.state_dict(), checkpoint_path)
else:
    m.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )

m.eval()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
