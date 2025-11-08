from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import pytorch_lightning as pl
from datasets import load_dataset
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from word2wec.data import Word2VecDataset
from word2wec.model import Word2VecRunner


def train(mode="cbow"):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    dataset = Word2VecDataset(ds["text"], window_size=2, mode=mode, num_negative=5)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=Word2VecDataset.collate_fn)

    runner = Word2VecRunner(dataset.vocab_size, embedding_dim=50, mode=mode)

    wandb_logger = WandbLogger(project="edu", name="w2v")
    trainer = pl.Trainer(max_epochs=10, enable_checkpointing=False, logger=wandb_logger,  # fast_dev_run=True
                         )

    trainer.fit(runner, loader)


if __name__ == "__main__":
    train(mode="cbow")
