import json
import pytorch_lightning as pl
import torch
from datasets import load_dataset
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from word2wec.data import Word2VecDataModule
from word2wec.model import Word2VecRunner


def train(mode="cbow", n_epoch=50, lr=0.025):
    pl.seed_everything(42, workers=True)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_text = dataset["train"]["text"]
    data_module = Word2VecDataModule(
        raw_text=train_text,
        batch_size=512,
        window_size=2,
        mode=mode,
        num_negative=10,
        min_count=5
    )
    print("Vocab size:", data_module.vocab_size)

    runner = Word2VecRunner(
        data_module.vocab_size,
        embedding_dim=200,
        mode=mode,
        n_steps=n_epoch*len(data_module.train_dataloader()),
        lr=lr
    )

    trainer = pl.Trainer(
        max_steps=n_epoch*len(data_module.train_dataloader()),
        enable_checkpointing=False,
        logger=WandbLogger(project="edu", name="w2v"),
        gradient_clip_val=1.0,
        callbacks=[LearningRateMonitor(logging_interval='step')],
        # detect_anomaly=True
        # fast_dev_run=True
    )

    trainer.fit(runner, data_module)

    torch.save(runner.model.state_dict(), "word2vec_weights.pt")
    with open("word2idx.json", "w") as f:
        json.dump(data_module.word2idx, f)


if __name__ == "__main__":
    train(mode="cbow")
