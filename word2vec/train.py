import json
import pytorch_lightning as pl
import torch
from datasets import load_dataset
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from data import Word2VecDataModule
from model import Word2VecRunner
from test import examples

analogy_examples = [
    # Gender / titles
    ("king", "man", "woman", "queen"),
    ("prince", "man", "woman", "princess"),
    ("actor", "man", "woman", "actress"),
    ("waiter", "man", "woman", "waitress"),
    ("hero", "man", "woman", "heroine"),
    ("duke", "man", "woman", "duchess"),
    ("father", "man", "woman", "mother"),
    ("husband", "man", "woman", "wife"),
    ("son", "man", "woman", "daughter"),
    ("brother", "man", "woman", "sister"),

    # Pluralization (plural - singular + other_singular = other_plural)
    ("cars", "car", "dog", "dogs"),
    ("cats", "cat", "bird", "birds"),
    ("children", "child", "mouse", "mice"),
    ("feet", "foot", "teeth", "tooth"),
    ("leaves", "leaf", "wolves", "wolf"),
    ("people", "person", "geese", "goose"),
    ("books", "book", "pages", "page"),
    ("cities", "city", "countries", "country"),
    ("apples", "apple", "oranges", "orange"),
    ("houses", "house", "rooms", "room"),

    # Verb progressive/continuous (ing)
    ("running", "run", "swim", "swimming"),
    ("eating", "eat", "drink", "drinking"),
    ("writing", "write", "reading", "reading"),
    ("driving", "drive", "riding", "riding"),
    ("singing", "sing", "dancing", "dancing"),
    ("sleeping", "sleep", "working", "working"),
    ("cooking", "cook", "baking", "baking"),
    ("walking", "walk", "jogging", "jog"),
    ("playing", "play", "studying", "study"),
    ("flying", "fly", "swimming", "swim"),

    # Comparative (-er)
    ("bigger", "big", "small", "smaller"),
    ("faster", "fast", "slow", "slower"),
    ("stronger", "strong", "weak", "weaker"),
    ("longer", "long", "short", "shorter"),
    ("higher", "high", "low", "lower"),
    ("younger", "young", "old", "older"),
    ("richer", "rich", "poor", "poorer"),
    ("brighter", "bright", "dark", "darker"),
    ("happier", "happy", "sad", "sadder"),
    ("nearer", "near", "far", "farther"),

    # Past tense
    ("ate", "eat", "go", "went"),
    ("ran", "run", "swim", "swam"),
    ("took", "take", "give", "gave"),
    ("wrote", "write", "think", "thought"),
    ("sat", "sit", "stand", "stood"),

    # Countries → capitals (capital - country + other_country = other_capital)
    ("paris", "france", "italy", "rome"),
    ("berlin", "germany", "spain", "madrid"),
    ("moscow", "russia", "japan", "tokyo"),
    ("ottawa", "canada", "brazil", "brasilia"),
    ("lisbon", "portugal", "ireland", "dublin"),

    # Part → whole relations
    ("hand", "finger", "toe", "foot"),
    ("crown", "king", "soldier", "helmet"),
    ("seed", "plant", "bird", "egg"),

    # Time relations / seasons
    ("morning", "dawn", "dusk", "evening"),

    # Semantic analogies
    ("teacher", "school", "hospital", "doctor"),

]

def train(mode="cbow", n_epoch=10, lr=0.015):
    pl.seed_everything(42, workers=True)

    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    train_text = dataset["train"]["text"]
    data_module = Word2VecDataModule(
        raw_text=train_text,
        batch_size=4096,
        window_size=5,
        mode=mode,
        num_negative=10,
        min_count=5,
        chunk_size=100,
        num_workers=12,
    )
    print("Vocab size:", data_module.vocab_size)

    runner = Word2VecRunner(
        data_module.vocab_size,
        word2idx=data_module.word2idx,
        idx2word=data_module.idx2word,
        examples=analogy_examples,
        embedding_dim=320,
        mode=mode,
        lr=lr,
    )

    if torch.cuda.is_available():
        accelerator = "gpu"
    elif torch.backends.mps.is_available():
        accelerator = "mps"
    else:
        accelerator = "cpu"

    trainer = pl.Trainer(
        max_steps=n_epoch*10000,
        limit_train_batches=10000,
        limit_val_batches=1000,
        enable_checkpointing=True,
        logger=WandbLogger(project="edu", name=f"w2v {mode}"),
        # gradient_clip_val=1.0,
        callbacks=[LearningRateMonitor(logging_interval='step')],
        accelerator=accelerator,
        # val_check_interval=10000,
        # detect_anomaly=True
        # fast_dev_run=True
    )

    trainer.fit(runner, data_module)

    torch.save(runner.model.state_dict(), f"{mode}_word2vec_weights.pt")
    with open(f"{mode}_word2idx.json", "w") as f:
        json.dump(data_module.word2idx, f)


if __name__ == "__main__":

    # train(mode="skipgram", n_epoch=20, lr=0.01)
    train(mode="cbow", n_epoch=20, lr=0.01)
