import json

import torch

from word2wec.model import Word2Vec

examples = [
    ("king", "man", "woman"),
    ("prince", "man", "woman"),
    ("paris", "france", "italy"),
    ("rome", "italy", "france"),
    ("walking", "walk", "running"),
    ("big", "bigger", "small"),
]

def analogy(word2idx, idx2word, emb_norm, a, b, c, topk=5):
    for w in (a,b,c):
        if w not in word2idx:
            raise KeyError(f"{w} not in vocab")
    vec = emb_norm[word2idx[a]] - emb_norm[word2idx[b]] + emb_norm[word2idx[c]]
    vec = vec / (vec.norm() + 1e-8)
    sims = torch.matmul(emb_norm, vec)
    topk_ids = sims.topk(topk + 3).indices.tolist()
    results = []
    for idx in topk_ids:
        w = idx2word[idx]
        if w in (a,b,c): continue
        results.append((w, float(sims[idx])))
        if len(results) >= topk:
            break
    return results

def test():
    with open("word2idx.json") as f:
        word2idx = json.load(f)
    idx2word = {int(i): w for w, i in word2idx.items()}
    model = Word2Vec(vocab_size=len(word2idx), embedding_dim=100, mode="cbow")
    state = torch.load("word2vec_weights.pt", map_location="cpu")
    model.load_state_dict(state)
    embeddings = model.in_embed.weight.detach().cpu()
    emb_norm = embeddings / (embeddings.norm(dim=1, keepdim=True) + 1e-8)

    print("\nAnalogy results (a - b + c -> ?):")
    for a, b, c in examples:
        try:
            print(f"{a} - {b} + {c} => {analogy(word2idx, idx2word, emb_norm, a, b, c)}")
        except KeyError as e:
            print(e)

if __name__ == "__main__":
    test()