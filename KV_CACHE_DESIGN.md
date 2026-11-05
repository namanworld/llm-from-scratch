# KV Cache in `bigram.py`: Design and Walkthrough

This guide explains the character-level Transformer and its key/value (KV) cache in `bigram.py`. It follows the current implementation and focuses on the data types, tensor shapes, mathematical operations, and generation flow.

The cache is an inference optimization: it avoids recomputing attention keys and values for old tokens while generating new tokens. It does not change the model's learned parameters or the training objective.

## 1. Configuration and notation

The current model settings in `bigram.py` are:

| Name | Value | Meaning |
|---|---:|---|
| `block_size` | 256 | Maximum context length |
| `n_embd` | 192 | Hidden/embedding dimension, also written \(d_\text{model}\) |
| `n_head` | 4 | Number of attention heads |
| `head_size` | 48 | Dimension of one head, \(d_\text{head}=192/4\) |
| `n_layer` | 4 | Number of Transformer blocks |
| `batch_size` | 32 | Training batch size |
| `vocab_size` | From `input.txt` | Number of unique characters |

Notation used below:

- \(B\): batch size
- \(T\): number of tokens in the current input
- \(P\): number of already cached tokens
- \(S=P+T\): total number of keys/values available to attend to
- \(d_\text{model}=192\): hidden dimension
- \(H=4\): number of heads
- \(d_\text{head}=48\): dimension per head
- \(V\): vocabulary size

For generation, the common case is \(B=1\). On the first call, \(T\) is the prompt length. On incremental calls, \(T=1\).

## 2. Data path: text to model input

This project uses a character vocabulary:

```python
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(text: str):
    return [stoi[x] for x in text]

def decode(code: list[int]):
    return "".join(itos[i] for i in code)
```

An input string becomes integer character IDs. For example, `"cat"` might encode as `[12, 1, 20]`; the exact IDs depend on the input file.

The embedding lookup maps IDs to vectors:

```python
tok_emb = self.token_embedding_table(idx)
```

Shapes:

```text
idx:     (B, T)             integer character IDs
tok_emb: (B, T, d_model)    here (B, T, 192)
```

The model adds a learned position vector to each token vector:

```python
positions = torch.arange(past_length, past_length + T, device=idx.device)
pos_emb = self.position_embedding_table(positions)
x = tok_emb + pos_emb
```

Shapes:

```text
positions: (T,)
pos_emb:   (T, d_model)
x:         (B, T, d_model)
```

PyTorch broadcasts `pos_emb` across the batch dimension when computing the sum.

Position matters: the same character at different sequence positions gets a different combined representation. During cached generation, `past_length` offsets the position of each new token so it does not repeatedly receive position zero.

## 3. One attention head: projections and shapes

Each `Head` receives a hidden-state tensor:

```text
x: (B, T, d_model)
```

It projects the hidden states into queries, keys, and values:

```python
k_new = self.key(x)
v_new = self.value(x)
q = self.query(x)
```

Each projection is a learned linear map from 192 dimensions to 48 dimensions:

```text
Wq, Wk, Wv: (d_model, d_head) = (192, 48)
q, k_new, v_new: (B, T, d_head) = (B, T, 48)
```

For a row-vector convention, the projection is:

\[
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
\]

where \(X\) has shape \((B,T,192)\), each weight has shape \((192,48)\), and each result has shape \((B,T,48)\). Biases are disabled in these three projections.

The \((192,48)\) shapes above describe the effective matrices in the equations. PyTorch stores an `nn.Linear(192, 48).weight` tensor with shape `(48, 192)` and applies the equivalent transposed multiplication internally.

### 3.1 Attention scores

The query at each position is compared with keys at all allowed positions:

```python
wei = q @ k.transpose(-2, -1) * self.head_size**-0.5
```

Shapes:

```text
q:                         (B, T, 48)
k.transpose(-2, -1):       (B, 48, S)
q @ k.transpose(-2, -1):   (B, T, S)
wei:                       (B, T, S)
```

The score for query position \(i\) and key position \(j\) is:

\[
\operatorname{score}_{ij}
=\frac{q_i\cdot k_j}{\sqrt{d_\text{head}}}
=\frac{q_i\cdot k_j}{\sqrt{48}}
\]

The scale is the **per-head** dimension, 48, not the combined model dimension, 192. Without this scaling, dot products tend to grow with vector dimension and can make softmax overly concentrated.

### 3.2 Causal mask and softmax

When processing a full sequence without an existing cache, later positions must not affect earlier positions. The lower-triangular mask sets scores for future keys to negative infinity:

```python
if cache is None:
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
wei = F.softmax(wei, dim=-1)
```

For a full sequence, the score tensor is \((B,T,T)\). A simplified causal mask for \(T=3\) is:

```text
allowed keys:
token 0: [yes,  no,  no ]
token 1: [yes,  yes, no ]
token 2: [yes,  yes, yes]
```

Softmax is applied along the key dimension (`dim=-1`). Each query's weights sum to one:

\[
A_{ij}=\operatorname{softmax}_j(\operatorname{score}_{ij})
\]

### 3.3 Weighted values and head output

The attention weights select and combine value vectors:

```python
out = wei @ v
```

Shapes:

```text
wei:        (B, T, S)
v:          (B, S, 48)
out:        (B, T, 48)
```

Mathematically:

\[
\operatorname{Head}(X)=\operatorname{softmax}
\left(\frac{QK^\mathsf{T}}{\sqrt{d_\text{head}}}+\text{mask}\right)V
\]

The mask term is omitted for incremental one-token calls because their key sequence contains only past and current tokens.

## 4. What is cached, and why

For every attention head, the cache is a tuple:

```python
(k, v)
```

Each tensor has shape:

```text
k: (B, cached_sequence_length, d_head)
v: (B, cached_sequence_length, d_head)
```

Queries are not cached. A query is the current token's request for information and is used in the current attention operation only. Old tokens' keys and values are repeatedly needed by future tokens, so those are stored.

When a new token arrives, the head computes its new key and value, then appends them along the sequence axis (`dim=1`):

```python
k = torch.cat((past_k, k_new), dim=1)
v = torch.cat((past_v, v_new), dim=1)
```

For example:

```text
past_k: (1, 5, 48)
k_new:  (1, 1, 48)
new k:  (1, 6, 48)
```

The same applies to values. The new query has shape `(B, 1, 48)`. It attends over all cached and current keys:

```text
q:                 (B, 1, 48)
k.transpose(...):  (B, 48, P+1)
scores:            (B, 1, P+1)
v:                 (B, P+1, 48)
head output:       (B, 1, 48)
```

There is no future key in this cache: it contains only past tokens plus the current token. Therefore, a one-token query can attend to the whole cache without applying a triangular mask.

## 5. Multi-head attention

`MultiHeadAttention` owns four independent `Head` modules. Each head has its own projection weights and its own KV cache.

For each head, the outputs are `(B,T,48)`. Concatenation joins the feature dimension:

```python
out = torch.cat(head_outputs, dim=-1)
```

Shapes:

```text
head 0 output: (B, T, 48)
head 1 output: (B, T, 48)
head 2 output: (B, T, 48)
head 3 output: (B, T, 48)
concatenated:  (B, T, 192)
```

The output projection mixes information across heads:

```python
out = self.dropout(self.proj(out))
```

Its linear map has shape `(192, 192)`, so the result remains `(B,T,192)`.

The cache returned by multi-head attention is a list with one `(k,v)` tuple per head:

```text
[
  (head_0_k, head_0_v),
  (head_1_k, head_1_v),
  (head_2_k, head_2_v),
  (head_3_k, head_3_v),
]
```

## 6. Transformer block

Each block applies pre-normalized self-attention and a feed-forward network, with residual connections:

```python
out, new_caches = self.sa(self.ln1(x), cache)
x = x + out
x = x + self.ffwd(self.ln2(x))
return x, new_caches
```

### Attention sublayer

```text
x:                      (B, T, 192)
ln1(x):                 (B, T, 192)
multi-head attention:   (B, T, 192)
x + attention output:  (B, T, 192)
```

The residual addition requires both operands to have the same shape.

### Feed-forward sublayer

The feed-forward network acts independently on each token position:

```python
nn.Linear(192, 4 * 192)  # 192 -> 768
nn.ReLU()
nn.Linear(768, 192)      # 768 -> 192
nn.Dropout(dropout)
```

Shapes:

```text
input:                 (B, T, 192)
first linear:          (B, T, 768)
ReLU:                  (B, T, 768)
second linear:         (B, T, 192)
dropout and residual:  (B, T, 192)
```

The feed-forward network has no KV cache. During incremental generation, it only needs to process the newest position; the attention cache avoids recomputing old keys and values.

## 7. Cache structure across layers

`BigramLanguageModel` uses a `ModuleList` so it can explicitly pass each block its own cache and collect the updated one.

There are four blocks, and each block has four heads. The nested structure returned by `forward` is:

```text
new_caches[block_index][head_index] == (key_tensor, value_tensor)
```

For a one-item batch and cache length \(S\):

```text
new_caches: list of 4 blocks
  each block cache: list of 4 heads
    each head cache:
      key:   (1, S, 48)
      value: (1, S, 48)
```

Caches must be separate by block and head. Different blocks receive different hidden states; different heads use different learned projections.

## 8. Model `forward`: prefill and incremental calls

The current method has the signature:

```python
forward(self, idx, targets=None, caches=None)
```

It returns:

```python
(logits, loss, new_caches)
```

### First call: prompt prefill

When `caches is None`:

1. `past_length` is zero.
2. The prompt's token and position embeddings are computed.
3. Every block processes the full prompt using its causal mask.
4. Each head returns keys and values for all prompt positions.

If prompt length is \(T=5\), each head's initial cache has shape:

```text
k, v: (B, 5, 48)
```

The prompt pass computes all prompt logits. The last prompt logit is used to sample the first output token.

### Later calls: one-token decode

When caches are supplied:

1. `past_length` is read from the first block's first head key tensor.
2. The incoming token gets position `past_length`.
3. Each head appends its new key and value to its cache.
4. The new token's query attends to all available cached keys and values.
5. Each block returns updated caches.

For \(P=5\), one token gives a new cache length of six:

```text
input idx:  (B, 1)
input x:    (B, 1, 192)
new k, v:   (B, 1, 48) per head before append
updated k,v:(B, 6, 48) per head after append
logits:     (B, 1, V)
```

## 9. Generation sequence

The current `generate` method follows this order:

1. Crop an overlong prompt to its last `block_size` tokens.
2. Run the prompt once to get prompt logits and initialize all caches.
3. Sample a token from the last-position logits.
4. If more output is requested, run only that token through the model with the caches.
5. Repeat.
6. When the cache is full, rebuild it from the latest `block_size` token window.

Sampling uses:

```python
probs = F.softmax(logits, dim=-1)
idx_next = torch.multinomial(probs, num_samples=1)
```

Shapes:

```text
logits[:, -1, :]: (B, V)
probs:            (B, V)
idx_next:         (B, 1)
idx after append: (B, original_length + generated_length)
```

The cache-full rebuild is important for this model's sliding context behavior. The positional embedding table has only `block_size` positions. Once 256 tokens are cached, the next context is recomputed from the latest 256 token IDs, with positions starting again at zero. This also bounds cache memory.

The model must be in evaluation mode for generation so dropout is disabled:

```python
m.eval()
generated = m.generate(context, max_new_tokens=500)
```

The script at the bottom of `bigram.py` already calls `m.eval()` before generation.

## 10. Training is still uncached

Training uses full sequences so positions can be processed in parallel:

```text
idx, targets: (B, T)
logits:       (B, T, V) before flattening
targets:      (B, T)
```

Cross-entropy expects a row of class scores per target, so the code reshapes:

```python
logits = logits.view(B * T, V)
targets = targets.view(B * T)
loss = F.cross_entropy(logits, targets)
```

The causal mask ensures token position \(t\) can only use positions \(\le t\). The forward call also returns caches, but the training loop ignores them:

```python
logits, loss, _ = m(xb, yb)
```

The cache is not reused across optimizer steps. Parameters change after every optimizer update, making previously computed keys and values stale; training also needs gradients through the full sequence.

## 11. Why caching saves work

Without a KV cache, generating one token at a time repeatedly processes the entire current context. For a context of length \(L\), each new token recomputes projections and attention for old positions.

With the cache:

- Prompt prefill still processes the prompt once.
- Each later step projects only the new token into \(q,k,v\).
- The new query still compares against all cached keys, so attention for that step grows with cache length.
- Old keys and values are not projected again.

Ignoring feed-forward and projection constants, full-context attention over \(L\) positions is quadratic in sequence length, \(O(L^2)\). Incremental cached attention still has to read prior keys and values, so generating \(N\) tokens with a cache of maximum length \(L\) costs roughly \(O(NL)\) attention-score/value work after prefill, rather than repeatedly recomputing \(O(L^2)\) attention at each step.

The current implementation concatenates tensors when extending the cache. That is clear and suitable for learning, but repeated `torch.cat` copies the existing cache each step. Production inference often uses preallocated cache buffers to avoid those copies.

## 12. Useful commands

Run these commands from the repository root.

### Set up dependencies

```bash
uv sync
```

### Check Python syntax without running training or generation

```bash
uv run python -m py_compile bigram.py
```

### Run the script

```bash
uv run python bigram.py
```

The script expects `input.txt` in the current working directory. With `train_model = True`, it trains and saves `model.pth` before generating. Set `train_model = False` after a checkpoint exists to load that checkpoint instead.

### Review the code changes

```bash
git diff -- bigram.py
git status --short
```

### Check selected runtime/device details

Add temporary prints while learning:

```python
print("device:", device)
print("x:", tuple(x.shape))
print("q:", tuple(q.shape))
print("k:", tuple(k.shape))
print("v:", tuple(v.shape))
print("attention scores:", tuple(wei.shape))
```

For an incremental step, expected shapes with cache length \(P\) are:

```text
x:                (B, 1, 192)
q:                (B, 1, 48)
k, v after append:(B, P + 1, 48)
wei:              (B, 1, P + 1)
head output:      (B, 1, 48)
```

Remove temporary prints after checking; printing every generation step can substantially slow generation.

## 13. Measuring generation latency

Two common metrics are:

- **TTFT (time to first token):** wall time from starting generation through prompt prefill and sampling the first token.
- **TPOT (time per output token):** mean time between subsequent generated tokens. For \(N>1\), average the \(N-1\) intervals after the first token.

Accelerator operations can be asynchronous. Synchronize before taking timestamps:

```python
import time

def synchronize_device():
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
```

Inside `generate`, start immediately before prompt prefill:

```python
synchronize_device()
start_time = time.perf_counter()
token_times = []
```

After sampling and appending each token, record:

```python
idx = torch.cat((idx, idx_next), dim=1)
synchronize_device()
token_times.append(time.perf_counter())
```

After the loop:

```python
if token_times:
    ttft = token_times[0] - start_time
    print(f"TTFT: {ttft:.4f} s")

    if len(token_times) > 1:
        intervals = [
            later - earlier
            for earlier, later in zip(token_times, token_times[1:])
        ]
        tpot = sum(intervals) / len(intervals)
        print(f"TPOT: {tpot:.4f} s/token")
```

These are end-to-end wall-clock timings and include sampling and Python overhead. Compare cache and no-cache runs with the same model, prompt, device, and number of output tokens. Run a warm-up first; otherwise startup effects can dominate a short measurement.

## 14. Implementation checklist

- [ ] `Head` caches only keys and values; queries are computed for the current input.
- [ ] Attention scores are scaled by \(1/\sqrt{d_\text{head}}\).
- [ ] The no-cache/full-sequence path applies a causal mask.
- [ ] A cached call processes only the newest token, so no future-token mask is needed.
- [ ] Every head has its own cache, and every block has its own collection of head caches.
- [ ] Cached token positions start at `past_length`.
- [ ] Generation runs in evaluation mode with gradients disabled.
- [ ] When cache length reaches `block_size`, the current code rebuilds from the latest context window.
- [ ] Training ignores returned caches and does not carry them across batches or optimizer updates.
- [ ] The model returns `(logits, loss, new_caches)`, so callers unpack three values.

## 15. Scope and limitations

This is an educational KV-cache implementation, not a fully optimized inference engine:

- Cache tensors grow with `torch.cat`, which copies prior cache contents.
- It supports the current single-token incremental-generation pattern; arbitrary chunked cached input would need a correctly offset causal mask.
- The sliding-window rebuild recomputes a full context when the cache reaches its maximum length.
- The implementation assumes caches are structurally compatible with the model and batch.
- It uses learned absolute positional embeddings limited to `block_size`.

These tradeoffs keep the code small enough to inspect while demonstrating the essential cache mechanics.
