import regex as re
text = """Hello, tokenizer! Unicode: café 😊"""
tokens = text.encode("utf-8")
tokens = list(map(int, tokens))

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


vocab_size = 276  # desired final vocabulary size
num_merges = vocab_size - 256
ids = list(tokens)


## BYTE PAIR ENCODING (BPE) ALGORITHM
merges = {}  # (int, int) -> int
for i in range(num_merges):
    counts = get_stats(ids)
    if not counts:
        break
    pair = max(counts, key=counts.get)
    idx = 256 + i
    print(f"Merging pair {pair} into new token {idx}")
    merges[pair] = idx
    ids = merge(ids, pair, idx)


# print("tokens length:", len(tokens))
# print("ids length:", len(ids))
# print(f"compression ratio: {len(tokens) / len(ids):.2f}")

# decoding
def decode(ids):
    tokens = b"".join(vocab[idx] for idx in ids)
    return tokens.decode("utf-8", errors="replace")

vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1] # byte addition

 
# encoding
def encode(text):
    tokens = list(text.encode("utf-8"))
    while len(tokens) > 1:
        counts = get_stats(tokens)
        pair = min(counts, key=lambda p: merges.get(p, float('inf')))
        if pair not in merges:
            break
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens


# force splits using regex pattern from GPT-2 tokenizer
gpt2pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
print(re.findall(gpt2pat, "Hello've world123 how's are you!!!?"))

    
