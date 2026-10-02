"""Chapter 11.4: context length and stride decide how many windows a text gives.

Run from `code/`:  python examples/ch11/stride_windows.py
"""

from llmfp.data import TokenWindowDataset

ids = list(range(20))
for context_length, stride in [(4, 4), (4, 2), (4, 1), (8, 8), (25, 25)]:
    dataset = TokenWindowDataset(ids, context_length, stride)
    starts = [dataset[i][0][0].item() for i in range(len(dataset))]
    print(f"context {context_length:>2}, stride {stride}: {len(dataset):>2} windows, starting at {starts}")

inputs, targets = TokenWindowDataset(ids, 4)[1]
print("\nwindow 1 (context 4, stride 4): inputs", inputs.tolist(), "targets", targets.tolist())
