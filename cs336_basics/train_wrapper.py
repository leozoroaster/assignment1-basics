import bpe_tokenizer
import lm_blocks
import optimizer
import data_process
import torch
from pathlib import Path
import numpy as np

from datasets import load_dataset
from multiprocessing import Pool
_tokenizer = None
_end_id = None

def init_worker(tokenizer_state):
    global _tokenizer, _end_id

    _tokenizer = tokenizer_state

    _end_id = _tokenizer.encode("<|endoftext|>")[0]

def process_line(line: str) -> np.ndarray:
    line = line.rstrip("\n")
    token_ids = _tokenizer.encode(line)
    token_ids.append(_end_id)  # one <|endoftext|> per story
    return np.array(token_ids, dtype=np.int32)

def train_model_TinyStories(d_model=512,h=16,d_ff=1344,vocab_size=10000,context_length=256,num_layers=4,theta=10000,raw_lr=1e-3,decay=1e-4,epoch_num=20,batch_num=256,batch_size=64,data_dir="data", save_dir="checkpoints",device=None):
    data_dir = Path(data_dir)
    save_dir = Path(save_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("start dataset loading")
    dataset=load_dataset("roneneldan/TinyStories")
    print("dataset loading successful")

    print("start text saving")
    text=dataset["train"]["text"]
    text_path = data_dir / "TinyStories.txt"
    token_path = data_dir / "TinyStories_tokens.int32"
    with open(text_path, "w", encoding="utf-8") as f:
        f.write("\n".join(text))
    print("text saving successful")

    vocab_file = data_dir / "TinyStories_vocab.txt"
    merges_file = data_dir / "TinyStories_merges.txt"

    if not (vocab_file.exists() and merges_file.exists()):
        print("Tokenizer not found — training BPE")
        tok = bpe_tokenizer.tokenizer_training(vocab_size, ["<|endoftext|>"])
        vocab, merges = tok.train_tokenizer(str(text_path))
        print("training BPE successful")

        bpe_tokenizer.tokenizer.save_tokenizer(vocab, merges, str(vocab_file), str(merges_file))
        print(f"saved tokenizer to {vocab_file} and {merges_file}")

    print("start tokenizing")
    tokenizer=bpe_tokenizer.tokenizer.from_files(
        vocab_filepath=str(vocab_file),
        merges_filepath=str(merges_file),
        special_tokens=["<|endoftext|>"],
    )
    print("tokenizer initiated")
    if __name__ == "__main__":
        tokenizer_state = tokenizer

        num_workers = 8  # use 8 vCPUs

        with open(text_path, "r", encoding="utf-8") as fin, \
                open(token_path, "wb") as fout, \
                Pool(processes=num_workers, initializer=init_worker, initargs=(tokenizer_state,)) as pool:

            for token_arr in pool.imap(process_line, fin, chunksize=64):
                token_arr.tofile(fout)

    raw_tokens = np.memmap(
        token_path,
        dtype=np.int32,
        mode="r"
    )
    print("tokenizing successful")
    print("num of tokens ", len(raw_tokens))
    print("tokens per epoch", batch_num*batch_size*context_length)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = device
    print("Using device:", device)
    epoch_num=epoch_num
    batch_num=batch_num

    print("creating LM")
    LM=lm_blocks.transformer_lm(d_model,h,d_ff,vocab_size,context_length,num_layers,theta).to(device)
    print("LM created")

    print("creating OPT")
    OPT=optimizer.AdamW(LM.parameters(), weight_decay=decay, lr=raw_lr)
    print("OPT created")

    print("start training")
    LM.train()

    for epoch in range(epoch_num):
        print("epoch ", epoch + 1)
        epoch_loss = 0.0
        for batch in range(batch_num):
            train_input, train_pred = data_process.data_loading(raw_tokens, batch_size, context_length, device_str=str(device))
            x = train_input.to(device)
            y = train_pred.to(device)
            OPT.zero_grad()
            logits = LM(x)
            loss_per_token = optimizer.cross_entropy(logits, y)
            loss = loss_per_token.mean()
            epoch_loss += loss.item()

            loss.backward()

            OPT.step(scheduler=(epoch,raw_lr,raw_lr*0.01,int(0.15*epoch_num),int(0.95*epoch_num)))
        print("epoch avg loss ", epoch_loss / batch_num)
    print("finished training")
    data_process.save_checkpoint(LM, OPT, epoch_num, save_dir / "train_LM_sanity_check.pt")
    print("finished model saving")

if __name__ == "__main__":
    train_model_TinyStories()