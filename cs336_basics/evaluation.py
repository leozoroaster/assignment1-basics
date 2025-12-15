import bpe_tokenizer
import lm_blocks
import optimizer
import data_process
import torch
from pathlib import Path
import numpy as np

def prepare_model(src, d_model=512,h=16,d_ff=1344,vocab_size=10000,context_length=256,num_layers=4,theta=10000,raw_lr=1e-3,decay=1e-4,epoch_num=100,batch_num=256,batch_size=64,data_dir="data", save_dir="checkpoints",device=None):
    LM=lm_blocks.transformer_lm(d_model,h,d_ff,vocab_size,context_length,num_layers,theta).to(device)
    OPT=optimizer.AdamW(LM.parameters(), weight_decay=decay, lr=raw_lr)

    data_process.load_checkpoint(src,LM,OPT, "cpu")

    return LM, OPT

def prepare_tokenizer(vocab_file,merges_file):
    tokenizer = bpe_tokenizer.tokenizer.from_files(
        vocab_filepath=str(vocab_file),
        merges_filepath=str(merges_file),
        special_tokens=["<|endoftext|>"],
    )
    return tokenizer

def evaluate_model(lm, tokenizer, prompts):
    answer=data_process.decode_prompt(lm, tokenizer, prompts)
    print(answer)

if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"

    checkpoint_path = DATA_DIR / "train_LM_sanity_check.pt"
    vocab_file = DATA_DIR / "TinyStories_vocab.txt"
    merges_file = DATA_DIR / "TinyStories_merges.txt"

    LM, OPT=prepare_model(checkpoint_path)
    tokenizer=prepare_tokenizer(vocab_file,merges_file)

    prompts=[
        "One day, Jack is sitting",
        "The red fox is eating",
        "My beautiful house",
        "Hello, is this",
        "I'm really ashamed"
    ]
    evaluate_model(LM,tokenizer,prompts)