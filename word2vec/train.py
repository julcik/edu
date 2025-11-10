import json
import pytorch_lightning as pl
import torch
from datasets import load_dataset
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from word2vec.data import Word2VecDataModule
from word2vec.model import Word2VecRunner
from word2vec.test import examples

analogy_examples = [
    ("king", "man", "woman", "queen"),
    ("prince", "man", "woman", "princess"),
    ("paris", "france", "italy", "rome"),
    ("rome", "italy", "france", "paris"),
    ("walking", "walk", "run", "running"),
    ("big", "bigger", "small", "smaller"),
    ("good", "better", "bad", "worse"),
    ("car", "cars", "dog", "dogs"),
    ("eat", "eating", "swim", "swimming"),
    ("long", "longer", "short", "shorter"),
]

def train(mode="cbow", n_epoch=10, lr=0.015):
    pl.seed_everything(42, workers=True)

    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    train_text = dataset["train"]["text"]
    data_module = Word2VecDataModule(
        raw_text=train_text,
        batch_size=512,
        window_size=5,
        mode=mode,
        num_negative=10,
        min_count=5
    )
    print("Vocab size:", data_module.vocab_size)

    runner = Word2VecRunner(
        data_module.vocab_size,
        word2idx=data_module.word2idx,
        idx2word=data_module.idx2word,
        examples=analogy_examples,
        embedding_dim=128,
        mode=mode,
        n_steps=n_epoch*len(data_module.train_dataloader()),
        lr=lr,

    )

    if torch.cuda.is_available():
        accelerator = "gpu"
    elif torch.backends.mps.is_available():
        accelerator = "mps"
    else:
        accelerator = "cpu"

    trainer = pl.Trainer(
        max_steps=n_epoch*len(data_module.train_dataloader()),
        enable_checkpointing=False,
        logger=WandbLogger(project="edu", name=f"w2v {mode} less noise"),
        # gradient_clip_val=1.0,
        callbacks=[LearningRateMonitor(logging_interval='step')],
        accelerator=accelerator,
        # detect_anomaly=True
        # fast_dev_run=True
    )

    trainer.fit(runner, data_module)

    torch.save(runner.model.state_dict(), f"{mode}_word2vec_weights.pt")
    with open(f"{mode}_word2idx.json", "w") as f:
        json.dump(data_module.word2idx, f)


if __name__ == "__main__":
    train(mode="skipgram")
    # train(mode="cbow", n_epoch=200)
