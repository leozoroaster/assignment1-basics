import torch
import numpy as np
import lm_blocks

def data_loading(x, batch_size, context_length, device_str="cpu"):
    L = len(x)

    assert L >= context_length+1, "Sequence too short for given context_length"

    # sample start positions
    starts = np.random.randint(0, L - context_length, size=batch_size)

    # allocate batch arrays in RAM (small)
    batch_inputs  = np.empty((batch_size, context_length), dtype=np.int64)
    batch_targets = np.empty((batch_size, context_length), dtype=np.int64)

    for i, s in enumerate(starts):
        # slice from memmap: this only reads the needed chunk from disk
        seq = x[s : s + context_length+1]  # length T+1
        batch_inputs[i]  = seq[:-1]
        batch_targets[i] = seq[1:]

    input_seqs = torch.from_numpy(batch_inputs).to(device_str)
    target_seqs = torch.from_numpy(batch_targets).to(device_str)

    return input_seqs, target_seqs

def save_checkpoint(model,optimizer,iteration,out):
  obj=dict()
  obj["model_weights"]=model.state_dict()
  obj["opt_state"] = optimizer.state_dict()
  obj["iteration"]=iteration
  torch.save(obj,out)

def load_checkpoint(src,model,optimizer, device=None):
    if device is not None:
        checkpoint = torch.load(src, map_location=torch.device(device))
    else:
        checkpoint = torch.load(src)

    # Load model weights
    model.load_state_dict(checkpoint['model_weights'])

    # Load optimizer state
    optimizer.load_state_dict(checkpoint['opt_state'])

    return checkpoint["iteration"]

def sample(v: torch.Tensor):
    v = v.cpu().numpy()
    p = np.random.rand()
    s = 0.0
    for i in range(len(v)):
        s += float(v[i])
        if s >= p:
            return i
    return len(v) - 1

def decode_prompt(LM,tokenizer,prompts,max_len=10,temperature=1,p=None, device=None):
  output_answers=[]
  for prompt in prompts:
    encoded_prompt = tokenizer.encode(prompt)
    generated_tokens=list(encoded_prompt)
    for _ in range(max_len):
      curr_prompt=generated_tokens
      input_tensor = torch.tensor(curr_prompt, dtype=torch.long, device=device).unsqueeze(0)
      with torch.no_grad():
            logits = LM(input_tensor)
      next_token_raw=logits[0,-1,:]

      next_token_prob=lm_blocks.softmax(next_token_raw/temperature)

      if p is not None:
        sorted_probs, sorted_indices = torch.sort(next_token_prob, descending=True)
        S = 0.0
        threshold = 0.0
        for i in range(len(sorted_probs)):
            S += float(sorted_probs[i])
            if S >= p:
                threshold = float(sorted_probs[i])
                break

        for i in range(len(next_token_prob)):
          if next_token_prob[i]<threshold:
            next_token_prob[i]=0

        next_token_prob=next_token_prob/next_token_prob.sum()

      next_token=sample(next_token_prob)

      generated_tokens.append(next_token)

      if tokenizer.decode([next_token])=="<|endoftext|>":
        break
    output_answers.append(tokenizer.decode(generated_tokens))
  return output_answers