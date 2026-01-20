# gpt2-rs

**GPT-2 inference implemented in pure Rust.**

![Demo](resources/demo.gif)

## Features

- **Zero ML dependencies** - Plain Rust, some basic file parsing and lots of arithmetic.
- **Single file** - The entire implementation lives in one 700-line file: tokenizer, 2D tensor, attention mechanism, inference loop
- **Easy to understand** - Focus was on making it easy to understand, not fast to run.

## Try it out!

```bash
# Download model weights from HuggingFace
# https://huggingface.co/openai-community/gpt2/

wget https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors -P resources/
wget https://huggingface.co/openai-community/gpt2/resolve/main/tokenizer.json -P resources/

cargo run --release
```

## Comparison with reference implementation

```python main.py
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model.eval()

prompt = "Life is "
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    output = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=25,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

print(tokenizer.decode(output[0]))
```

```bash
> uv run main.py

Life is  a very important part of our lives. We are all born with a certain amount of energy and we are all born with

```
