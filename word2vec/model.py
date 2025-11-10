import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

from word2vec.metric import AnalogyMetric


class Word2Vec(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim=50, mode="skipgram"):
        super().__init__()
        self.mode = mode
        self.in_embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)#, max_norm=1)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)#, max_norm=1)

    def forward(self, x, out):
        if self.mode == "skipgram":
            # predict context from center
            in_emb = self.in_embed(x)  # [B, D]
        else:
            # predict center from average of context embeddings
            context_emb = self.in_embed(x)  # [B, 2W, D]
            in_emb = context_emb.mean(dim=1)  # [B, D]

        out_emb = self.out_embed(out)  # [B, 1+N, D]
        return in_emb, out_emb


class Word2VecRunner(pl.LightningModule):
    def __init__(self, vocab_size, word2idx, idx2word, examples, embedding_dim=50, mode="skipgram", lr=0.01, n_steps=10000):
        super().__init__()
        self.save_hyperparameters(ignore=["word2idx", "idx2word", "examples"])
        self.model = Word2Vec(vocab_size=vocab_size, embedding_dim=embedding_dim, mode=mode)
        self.analogy_metric = AnalogyMetric(word2idx, idx2word, examples)

    def _step(self, batch, batch_idx):
        # stack target + negatives
        out = torch.cat([batch["target"].unsqueeze(1), batch["negatives"]], dim=1)  # [B, 1+N]

        in_emb, out_emb = self.model(batch["input"], out)

        pos_emb = out_emb[:, 0, :]  # [B, D]
        neg_emb = out_emb[:, 1:, :]  # [B, N, D]

        pos_score = torch.sum(in_emb * pos_emb, dim=1)  # [B]
        neg_score = torch.bmm(neg_emb, in_emb.unsqueeze(2)).squeeze(2)  # [B, N]

        # Cos sim with negative sampling
        # pos_score = torch.cosine_similarity(in_emb, pos_emb, dim=1)
        # neg_score = torch.bmm(
        #     torch.nn.functional.normalize(neg_emb, dim=2),
        #     torch.nn.functional.normalize(in_emb.unsqueeze(2), dim=1)
        # ).squeeze(2)

        loss = -F.logsigmoid(pos_score) - torch.sum(F.logsigmoid(-neg_score), dim=1)
        loss = loss.mean()
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        if self.hparams.mode == "skipgram":
            embeddings = self.model.in_embed.weight.detach().cpu()
        else:
            embeddings = self.model.out_embed.weight.detach().cpu()
        self.analogy_metric.update(embeddings)
        metrics = self.analogy_metric.compute()
        self.log_dict(metrics, prog_bar=True)
        self.analogy_metric.reset()

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optim,
            max_lr=self.hparams.lr,
            total_steps=self.hparams.n_steps,
            pct_start=0.05,
            final_div_factor=1e2
        )
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "step",
            },
        }
