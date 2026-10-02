"""Chapter 11.7: packing documents into full windows instead of padding each one.

Run from `code/`:  python examples/ch11/packing.py
"""

from llmfp.counting_lm import read_lines
from llmfp.data import pack_documents, pad_batch
from llmfp.tokenizers import load_tokenizer

tokenizer = load_tokenizer("data/tokenizer/harbor-bpe-2048.json")
separator = tokenizer.special_tokens["<|endoftext|>"]
documents = [tokenizer.encode(line) for line in read_lines("data/tiny/harbor.txt")]
lengths = [len(d) for d in documents]
print(f"{len(documents)} documents, {min(lengths)} to {max(lengths)} tokens each, {sum(lengths)} tokens in total")

padded = pad_batch(documents, pad_id=separator)
real = int(padded.attention_mask.sum())
print(f"padding: one row per document, shape {tuple(padded.input_ids.shape)}, "
      f"{padded.input_ids.numel() - real} of {padded.input_ids.numel()} positions are padding")

windows, owners = pack_documents(documents, context_length=32, separator_id=separator)
print(f"packing: shape {tuple(windows.shape)}, no padding; each window holds parts of several documents")
print("\nfirst packed window:", repr(tokenizer.decode(windows[0].tolist())))
print("document of each token:", owners[0].tolist())
