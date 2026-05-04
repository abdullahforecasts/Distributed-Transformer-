

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(42)

# ================= CONFIG =================
vocab_size    = 27        # a-z + '~'
embedding_dim = 32
hidden_dim    = 64
block_size    = 8

# ================= VOCAB =================
itos = [chr(ord('a') + i) for i in range(26)] + ['~']
stoi = {c: i for i, c in enumerate(itos)}

def encode(c): return stoi[c]
def decode(i): return itos[i]

# ================= MODEL =================
class MiniTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding    = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.Wq   = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk   = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv   = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W1   = nn.Linear(embedding_dim, hidden_dim,    bias=False)
        self.W2   = nn.Linear(hidden_dim,    embedding_dim, bias=False)
        self.Wout = nn.Linear(embedding_dim, vocab_size,    bias=False)

    def forward(self, tokens):
        T   = tokens.size(1)
        pos = torch.arange(T)
        x   = self.token_embedding(tokens) + self.position_embedding(pos)

        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        scores = Q @ K.transpose(-2, -1)
        mask   = torch.triu(torch.ones(T, T), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        scores = F.softmax(scores, dim=-1)
        out    = scores @ V
        out    = F.layer_norm(out + x, [embedding_dim])

        y = F.relu(self.W1(out))
        y = self.W2(y)
        y = F.layer_norm(y + out, [embedding_dim])

        return self.Wout(y)


# ================= SAVE WEIGHTS =================
def save_weights(model):
    def save(name, tensor):
        np.savetxt(name + ".txt", tensor.detach().numpy().flatten())

    save("token_embedding.weight",    model.token_embedding.weight)
    save("position_embedding.weight", model.position_embedding.weight)

    # Transpose Linear weights: PyTorch [out, in] -> C++ expects [in, out]
    save("Wq.weight",   model.Wq.weight.T)
    save("Wk.weight",   model.Wk.weight.T)
    save("Wv.weight",   model.Wv.weight.T)
    save("W1.weight",   model.W1.weight.T)
    save("W2.weight",   model.W2.weight.T)
    save("Wout.weight", model.Wout.weight.T)

    print("Weights saved.")


# ================= GENERATE =================
def generate(model, start_char, steps=1000):
    model.eval()
    tokens = [encode(start_char)]
    result = []

    with torch.no_grad():
        for _ in range(steps):
            ctx    = tokens[-block_size:]
            inp    = torch.tensor([ctx])
            logits = model(inp)
            probs  = F.softmax(logits[0, -1, :], dim=-1)
            nxt    = torch.argmax(probs).item()
            tokens.append(nxt)
            result.append(decode(nxt))

    return ''.join(result)


# ================= MAIN =================
if __name__ == "__main__":
    model = MiniTransformer()
    save_weights(model)

    # --- Single pass verification for each starting token ---
    test_starts = ['~', 'a', 'm', 'z']

    print("=" * 60)
    print("SINGLE PASS VERIFICATION (first predicted char + probs)")
    print("=" * 60)

    for start in test_starts:
        tokens_in = torch.tensor([[encode(start)]])
        with torch.no_grad():
            logits = model(tokens_in)
        probs  = F.softmax(logits[0, -1, :], dim=-1)
        pred   = torch.argmax(probs).item()
        print(f"Start='{start}' | predicted='{decode(pred)}' | "
              f"probs={[round(p.item(),5) for p in probs]}")

    print()

    # --- 1000-char generation from multiple starts ---
    print("=" * 60)
    print("1000-CHAR GENERATION")
    print("=" * 60)

    for start in test_starts:
        stream = generate(model, start, steps=1000)
        print(f"\nStart='{start}':")
        # Print in lines of 100 for readability
        for i in range(0, len(stream), 100):
            print(stream[i:i+100])