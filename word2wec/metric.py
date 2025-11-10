import torch
from torchmetrics import Metric

class AnalogyMetric(Metric):
    full_state_update = False
    def __init__(self, word2idx, idx2word, examples, topk=10):
        super().__init__()
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.examples = examples
        self.topk = topk
        self.add_state("mrr", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("hits", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, embeddings: torch.Tensor):
        emb = embeddings / (embeddings.norm(dim=1, keepdim=True) + 1e-8)
        vocab_size = emb.size(0)

        for a, b, c, expected in self.examples:
            if any(w not in self.word2idx for w in (a, b, c, expected)):
                continue
            vec = emb[self.word2idx[a]] - emb[self.word2idx[b]] + emb[self.word2idx[c]]
            vec = vec / (vec.norm() + 1e-8)
            sims = torch.matmul(emb, vec)
            topk_ids = sims.topk(self.topk).indices.tolist()
            ranks = torch.argsort(sims, descending=True)
            rank = (ranks == self.word2idx[expected]).nonzero(as_tuple=True)[0]
            rank = int(rank) + 1 if len(rank) > 0 else vocab_size

            # update metrics
            self.mrr += 1.0 / rank
            if self.word2idx[expected] in topk_ids:
                self.hits += 1.0
            self.count += 1

    def compute(self):
        return {
            "analogy_mrr": self.mrr / self.count,
            "analogy_hits@k": self.hits / self.count
        }
