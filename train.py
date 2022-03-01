import argparse
import random
import numpy as np
import torch
import pytorch_lightning as pl

from model import T5FineTuner

# based on: https://colab.research.google.com/github/sonoisa/t5-japanese/blob/main/t5_japanese_classification.ipynb#scrollTo=H_Fzhe5qFV3J

# Setting the random number seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# GPU usage
USE_GPU = torch.cuda.is_available()

# Hyper parameters
args_dict = dict(
    data_dir="./dataset/",  # Directory of datasets
    model_name_or_path="google/mt5-small",
    tokenizer_name_or_path="google/mt5-small",
    save_pretrained_path="./models/",

    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    gradient_accumulation_steps=1,

    n_gpu=1 if USE_GPU else 0,
    early_stop_callback=False,
    fp_16=False,
    opt_level='O1',
    max_grad_norm=1.0,
    seed=42,
)

# Hyper parameters for learning
args_dict.update({
    "max_input_length":  512,  # Maximum number of tokens in input
    "max_target_length": 4,  # Maximum number of tokens in output
    "train_batch_size":  8,
    "eval_batch_size":   8,
    "num_train_epochs":  4,
    })
args = argparse.Namespace(**args_dict)

train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    precision= 16 if args.fp_16 else 32,
    amp_backend='apex',
    gradient_clip_val=args.max_grad_norm,
)

def train():
    # fine tuning
    model = T5FineTuner(args)
    trainer = pl.Trainer(**train_params)
    trainer.fit(model)

    # save the tokenizer/model for the last epoc
    model.tokenizer.save_pretrained(args.save_pretrained_path)
    model.model.save_pretrained(args.save_pretrained_path)


if __name__ == '__main__':
    set_seed(42)
    train()