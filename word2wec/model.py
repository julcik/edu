import torch
from torch import nn
import pytorch_lightning as pl


class Word2Vec(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim=50, mode="skipgram"):
        super().__init__()
        self.mode = mode
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)

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
    def __init__(self, vocab_size, embedding_dim=50, mode="skipgram", lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.model = Word2Vec(vocab_size=vocab_size, embedding_dim=embedding_dim, mode=mode)

    def training_step(self, batch, batch_idx):

        # stack target + negatives
        out = torch.cat([batch["target"].unsqueeze(1), batch["negatives"]], dim=1)  # [B, 1+N]

        in_emb, out_emb = self.model(batch["input"], out)
        # unstack
        pos_emb = out_emb[:, 0, :] # [B, D]
        neg_emb = out_emb[:, 1:, :] # [B, N, D]

        pos_score = torch.sum(in_emb * pos_emb, dim=1)  # [B]
        neg_score = torch.bmm(neg_emb, in_emb.unsqueeze(2)).squeeze(2)  # [B, N]

        # Dot product with negative sampling loss
        loss = -torch.log(torch.sigmoid(pos_score)) - torch.sum(torch.log(torch.sigmoid(-neg_score)), dim=1)
        loss = loss.mean()

        # Cos sim with negative sampling
        # pos_score = torch.cosine_similarity(in_emb, pos_emb, dim=1)
        # neg_score = torch.bmm(
        #     torch.nn.functional.normalize(neg_emb, dim=2),
        #     torch.nn.functional.normalize(in_emb.unsqueeze(2), dim=1)
        # ).squeeze(2)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
