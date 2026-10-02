## Chapter 11: Sequences, Batches, and Next-Token Targets

[Back to index](../../README.md) · Previous: [Chapter 10](ch10-embeddings-and-position.md) · Next: Chapter 12 (planned)

You now have a tokenizer that turns text into IDs (Chapter 9) and embeddings that turn IDs into vectors (Chapter 10). Between them sits a step that is easy to get subtly wrong: turning a corpus of token IDs into **batches of training examples** of a fixed shape, with the right targets, without wasting computation on padding, and without letting one document leak into another. Bugs in this step rarely raise errors. They produce models that train, report a decreasing loss, and learn the wrong thing.

This chapter builds the book's data module, the one Parts 3 and 4 use to train the transformer, and tests it thoroughly. It ends by training the simplest model that makes a prediction at every position of a sequence, which is exactly how a transformer is trained.

#### Learning outcomes

By the end of this chapter you will be able to:

1. Explain context windows and context length, and why one window of T tokens gives T training examples.
2. Build inputs and targets with the correct shift by one, and write a test that catches every off-by-one variant.
3. Choose a stride and explain the trade-off between overlap and repetition.
4. Batch variable-length sequences with padding, attention masks, and ignored targets, and verify that padding does not affect the loss.
5. Pack documents with separator tokens, and explain what packing gains and what it risks.
6. Use PyTorch's `Dataset` and `DataLoader`, and compute a loss over every position of a batch.

#### Prerequisites

- [Chapter 3](../part-1-foundations/ch03-tensors.md): slicing (`[:, :-1]`, `[:, 1:]`), `reshape`, `unfold`, batches.
- [Chapter 6](../part-1-foundations/ch06-how-training-works.md): cross-entropy loss, the training loop, minibatches.
- [Chapter 7.9](../part-1-foundations/ch07-project-0-char-model.md): the off-by-one target bug.
- [Chapter 9.6](ch09-byte-pair-encoding.md): special tokens and `allowed_special`.
- [Chapter 10.7](ch10-embeddings-and-position.md): position embeddings and the maximum sequence length.

#### New terms in this chapter

| Term | Plain-English meaning | Section |
|---|---|---|
| Context length | The number of tokens in one training window; also the most a model can take at once | 11.2 |
| Window | A fixed-length slice of the token stream used as one training sequence | 11.2 |
| Stride | How far the start of each window moves from the previous one | 11.4 |
| Padding, pad token | Filler tokens that make shorter sequences the same length as longer ones | 11.6 |
| Attention mask | A tensor marking which positions are real tokens and which are padding | 11.6 |
| Ignore index | A special target value (-100 in PyTorch) that the loss skips | 11.6 |
| Packing | Joining documents end to end, with a separator token, and cutting the stream into full windows | 11.7 |
| Separator (end-of-text) token | A special token marking where one document ends and the next begins | 11.7 |
| `Dataset`, `DataLoader` | PyTorch's interfaces for indexing examples and serving them in shuffled batches | 11.8 |

---

### 11.1 The problem: one long text must become many same-shaped examples

The harbor corpus is about 1,000 distinct sentences. A real pretraining corpus (Chapter 18) is millions of documents of every length, from a line to a book. A model, on the other hand, needs tensors of a fixed shape: a batch of B sequences, each exactly T tokens long, with a matching tensor of targets. Getting from one to the other raises several questions:

- How long should each training sequence be, and where should sequences start?
- Which token is the target at each position?
- What happens to documents shorter than T, or longer?
- How do we avoid computing on filler, and avoid letting one document influence predictions about another?

Each section answers one question, with code from the data module and a test.

---

### 11.2 Context windows and context length

A **window** is a fixed-length slice of the token stream. Its length, the **context length** T, is the number of tokens the model sees at once during training. It is a hyperparameter with direct costs and limits:

- **Memory and compute** grow with T. Every layer of a transformer processes a `(batch, T, d_model)` tensor, and attention (Chapter 13) compares every position with every earlier one, so its work grows faster than T itself.
- **The model's maximum.** A model with learned position embeddings has one per position up to its context length (Chapter 10.7). It cannot take longer sequences. Many published models state their context length in their configuration (Chapter 22.2).
- **What the model can use.** A pattern spanning more than T tokens cannot be learned from any single window. Chapter 7's "Before the storm." failure was a window too short to hold the start of its own sentence.

The crucial difference from Part 1: in Chapter 7, one window of 12 characters gave **one** training example, predicting the character after the window. A transformer predicts the next token at **every** position of the window at once, as section 11.3 shows. A window of T tokens gives T examples: predict token 2 from token 1, token 3 from tokens 1–2, and so on up to token T+1 from all T. This is much more efficient, and it means a window needs T+1 tokens: T inputs plus one more token as the last target.

File: [`code/llmfp/data/windows.py`](../../code/llmfp/data/windows.py)

```python
@@FILE code/llmfp/data/windows.py@@
```

`TokenWindowDataset` stores the whole stream of IDs once and computes each window on request. `__len__` reports how many complete windows fit; a window that would run past the end is dropped rather than padded.

---

### 11.3 Inputs and targets: the shift by one, and off-by-one errors

For a window, the inputs are tokens `start` to `start + T - 1`, and the targets are the same tokens shifted one position later: tokens `start + 1` to `start + T`. At every position, the target is the token that comes **next**.

File: [`code/examples/ch11/shift_by_one.py`](../../code/examples/ch11/shift_by_one.py)

```python
@@FILE code/examples/ch11/shift_by_one.py@@
```

Observed output:

```text
@@RUN python examples/ch11/shift_by_one.py@@
```

The correct version reads like a set of exercises: having seen "The keeper lit", predict " the". Each position sees only the tokens up to and including itself. Making sure a transformer *only* uses those tokens, and not later ones it can technically access, is the job of the causal mask in Chapter 13.6.

The wrong versions are each a real bug that has occurred in real code:

- **No shift.** The target is the token the model was just given. The model learns to copy its input, the loss drops toward zero, and the model is useless. This is Chapter 7.9's bug.
- **Shifted the wrong way.** The target is the *previous* token. The model learns to predict the past.
- **Shifted by two.** The model learns to skip a token. This one is insidious: the loss decreases, generated text looks plausible but strange, and nothing crashes.

All three are caught by one test on a hand-checkable sequence. `test_targets_are_inputs_shifted_by_one` in [`code/tests/test_data.py`](../../code/tests/test_data.py) builds windows over the IDs 100 to 119 (distinct values, so every position is identifiable) and checks that the targets equal the inputs shifted by one and that each window's last target is the token after the window. Every off-by-one variant fails it. Run this kind of test whenever you change data code.

---

### 11.4 Sliding windows and stride

The **stride** is how far each window's start moves from the previous window's start.

File: [`code/examples/ch11/stride_windows.py`](../../code/examples/ch11/stride_windows.py)

```python
@@FILE code/examples/ch11/stride_windows.py@@
```

Observed output:

```text
@@RUN python examples/ch11/stride_windows.py@@
```

- **Stride equal to the context length** (the default) gives windows that share exactly one token: the last target of one window is the first input of the next. Every next-token step in the stream is a target exactly once (a test checks this). This is the usual choice for pretraining.
- **A smaller stride** gives overlapping windows: more windows from the same text, and each token is predicted several times with different amounts of preceding context. It can help when data is scarce, at the cost of training on repeated material, which raises the risk of overfitting (Chapter 4.5).
- **A context longer than the text** gives no windows at all. The dataset reports zero rather than inventing padding.

A token near the start of a window has little context; one near the end has nearly T tokens of it. With non-overlapping windows, the model learns to predict from every amount of context, which is what it will face during generation (Chapter 17).

---

### 11.5 Batches

A batch stacks B windows into tensors of shape `(B, T)`: inputs and targets. One optimizer step therefore learns from B times T predictions: the batch size times the context length. That product, *tokens per batch*, is the number that matters for training dynamics and memory (Chapter 19.8), more than either factor alone.

Memory follows from Chapter 3.12. The IDs themselves are small, but the model turns every position into a `d_model`-sized vector in every layer, so activation memory grows with B, T, `d_model`, and the number of layers together. When a run does not fit in memory, reducing B (and using gradient accumulation, Chapter 6.8, to keep the effective batch size) is usually the first remedy; reducing T changes what the model can learn.

---

### 11.6 Variable lengths: padding, masks, and ignored targets

Sometimes examples must stay separate: one question and answer per row in instruction tuning (Chapter 27), one message per row in classification (Chapter 26). Separate examples have different lengths, and tensors cannot be ragged. The standard solution is **padding**: fill shorter sequences with a **pad token** up to the length of the longest, and record which positions are real.

File: [`code/llmfp/data/collate.py`](../../code/llmfp/data/collate.py)

```python
@@FILE code/llmfp/data/collate.py@@
```

`pad_batch` returns three tensors of the same shape:

- **`input_ids`**: the inputs, with `pad_id` in the filler positions.
- **`targets`**: the shifted targets, with **`IGNORE_INDEX`** (-100) wherever there is no real target. PyTorch's cross-entropy skips any target equal to its `ignore_index`, which is -100 by default, so padded positions contribute nothing to the loss or the gradients.
- **`attention_mask`**: `True` at real positions, `False` at padding. The loss does not need it (the ignore index already handles targets), but the model does: Chapter 13.7 uses it to stop real tokens from attending to padding.

File: [`code/examples/ch11/padding_masks.py`](../../code/examples/ch11/padding_masks.py)

```python
@@FILE code/examples/ch11/padding_masks.py@@
```

Observed output:

```text
@@RUN python examples/ch11/padding_masks.py@@
```

Three things to read in it:

- **Flattening for the loss.** The model produces logits of shape `(batch, length, vocab)`; cross-entropy expects one row per example. `reshape(-1, vocab_size)` turns every (row, position) pair into one example, and `targets.reshape(-1)` lines up with it. You will write this line in every language-model training loop.
- **Ignored padding changes nothing.** The loss over the padded batch with ignored targets equals the loss computed over the real positions alone. A test (`test_ignored_padding_does_not_change_the_loss`) guarantees it.
- **The bug: padding as a target.** If padded targets are set to the pad ID instead of the ignore index, five of the twelve positions train the model to predict padding, and the loss changes, here from 2.72 to 3.17. The model then learns that after short sequences comes padding, and wastes capacity on it. A further danger: if the pad ID is also a real token (here, 0 could be one), the model learns false facts about that token.

**Left or right padding?** For training, right padding (filler after the tokens) is natural. For generation (Chapter 17), many implementations prefer left padding, so that every sequence in a batch ends at the same position and the next token is predicted from the last column for all rows at once. `pad_batch(side="left")` provides it.

---

### 11.7 Packing several documents into one sequence; document boundaries

Padding wastes computation on filler. How much depends on how much lengths vary. **Packing** avoids it: join all documents into one stream with a **separator token** between them, then cut the stream into full windows.

File: [`code/examples/ch11/packing.py`](../../code/examples/ch11/packing.py)

```python
@@FILE code/examples/ch11/packing.py@@
```

Observed output:

```text
@@RUN python examples/ch11/packing.py@@
```

The 40 hand-written harbor sentences, padded one per row, waste 276 of 600 positions, almost half. Packed into windows of 32, they waste none.

The separator is the tokenizer's `<|endoftext|>` special token (Chapter 9.6). It does two jobs: it tells the model where documents end, so the model learns that after a separator a new, unrelated document begins (the bigram model below learned that sentences start with "The", "At", or "In"); and at generation time, producing it is a natural stop signal (Chapter 21.6). Code that builds the stream inserts the separator ID directly, so untrusted text that happens to contain the characters `<|endoftext|>` cannot create one.

Packing has a cost. A window can now contain the end of one document and the start of another, and the model, when predicting tokens of the second document, can see the first. For unrelated documents, that is noise the model must learn to ignore; for some tasks it can also leak information between examples. Three common positions:

1. **Accept it.** Many pretraining pipelines pack and rely on separators; the model learns that text before a separator is usually irrelevant.
2. **Mask across boundaries.** `pack_documents` also returns `document_ids`, recording which document each token came from. Chapter 13.7 uses it to build an attention mask that stops tokens from seeing earlier documents in the same window.
3. **Do not pack** where examples must stay strictly separate, such as evaluation sets and some fine-tuning data (Chapter 27.6).

Which to choose is an engineering decision; record it with the run.

#### A middle ground: bucketing

If examples must stay separate, padding waste can still be cut by putting examples of similar length in the same batch. Exercise 4's solution sorts by length, cuts batches, and shuffles the order of the batches:

```text
@@RUN python -m solutions.ch11_bucketing@@
```

Random batches of BPE-encoded harbor sentences waste over a fifth of their positions; length-bucketed batches waste almost nothing. The cost is that batches are less random (each contains similar examples), which can slightly affect training; shuffling the batch order keeps most of the benefit of shuffling.

---

### 11.8 PyTorch `Dataset` and `DataLoader`

PyTorch separates two jobs:

- A **`Dataset`** answers "how many examples are there?" (`__len__`) and "give me example i" (`__getitem__`). `TokenWindowDataset` is one; `TensorDataset` wraps ready-made tensors as one. These are the dunder methods from Chapter 2.6.
- A **`DataLoader`** draws examples from a dataset, optionally shuffled, groups them into batches, and stacks them into tensors.

File: [`code/examples/ch11/dataloader.py`](../../code/examples/ch11/dataloader.py)

```python
@@FILE code/examples/ch11/dataloader.py@@
```

Observed output:

```text
@@RUN python examples/ch11/dataloader.py@@
```

The settings that matter:

- **`batch_size`** and **`shuffle`**: shuffling gives a new order every epoch, as Chapter 6.7 required.
- **`generator`**: a seeded `torch.Generator` makes the shuffling reproducible, independently of other random draws in the program (Chapter 4.6).
- **`drop_last`**: whether to discard a final, smaller batch. Training code often drops it so every step has the same batch size; evaluation keeps it so every example is measured.
- **`num_workers`** (not used here): prepares batches in background processes, which helps when loading data is slow. It introduces its own reproducibility concerns, covered in Chapter 19.

---

### 11.9 Milestone: a tested data path, and a model that predicts at every position

The milestone joins everything: text, tokenizer, documents, packing, DataLoader, and a model trained on every position. The model is the simplest one with a transformer's interface: token IDs of shape `(batch, seq)` in, logits of shape `(batch, seq, vocab)` out.

File: [`code/llmfp/model/bigram.py`](../../code/llmfp/model/bigram.py)

```python
@@FILE code/llmfp/model/bigram.py@@
```

A **bigram model** predicts each next token from the current token only: its embedding table has one row per token, and the row is read directly as the logits for whatever follows. It is Chapter 1's one-word counting model, learned by gradient descent.

File: [`code/configs/batches-cpu.toml`](../../code/configs/batches-cpu.toml)

```toml
@@FILE code/configs/batches-cpu.toml@@
```

File: [`code/scripts/ch11_build_batches.py`](../../code/scripts/ch11_build_batches.py)

```python
@@FILE code/scripts/ch11_build_batches.py@@
```

```bash
python -m scripts.ch11_build_batches
```

Observed output (about 10 seconds on the test machine):

```text
@@RUN python -m scripts.ch11_build_batches@@
```

Read it from the top:

1. **Padding versus packing**, on BPE tokens of the deduplicated harbor training set: padding each sentence to the longest would waste about a quarter of positions; packing wastes none. With windows of 32 tokens and sentences of about 11, every window spans several documents.
2. **One batch.** Windows of 33 tokens become inputs and targets of 32, and the decoded tokens show the shift: the target row is the input row moved one place to the left. The first window starts mid-stream, right after a sentence ended, which is normal for packed data.
3. **Training.** The untrained model's loss is above the even-spread reference (8.36 versus 7.63): its random starting logits are not even, and unevenly wrong guesses cost more than even ones. Training brings the validation loss down to about 1.46.
4. **What it learned.** After `' at'`, the most likely next tokens are times of day; after `<|endoftext|>`, the words sentences start with. Like the counting model, it knows only one token of context, so it cannot know which actor's actions are allowed (Chapter 7.7's fluent-but-false sentences are exactly this failure).

`sequence_loss` contains the line Part 4's training loop is built around:

```python
nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
```

Everything in Part 3 replaces only the model. The data path stays.

---

### 11.10 Common mistakes, recap, concept checks, exercises, answers, and checkpoint

#### Common mistakes

| Symptom | Likely cause | Fix |
|---|---|---|
| Loss drops to near zero quickly; generation is degenerate | Targets not shifted (or shifted the wrong way) | Test the shift on distinct IDs |
| Loss decreases but text skips or garbles words | Shift by two, or windows misaligned with targets | Same test; inspect a decoded batch |
| `Expected input batch_size (X) to match target batch_size (Y)` in cross-entropy | Logits and targets flattened differently | `logits.reshape(-1, vocab)` and `targets.reshape(-1)` |
| Loss differs between padded and unpadded runs of the same data | Padding positions used as targets | Fill padded targets with -100 (`IGNORE_INDEX`) |
| Model tends to emit pad tokens | Same as above | Same |
| Zero windows, or an empty DataLoader | Context length longer than the data | Shorter context or more data; check `len(dataset)` |
| Different batches on every run | Shuffle without a seeded generator | Pass `generator=torch.Generator().manual_seed(seed)` |
| User text produces the end-of-text token | Separators created by encoding text with special tokens allowed | Insert separator IDs in code; encode untrusted text without `allowed_special` |
| Memory error during training | Too many tokens per batch | Smaller batch with gradient accumulation; then consider shorter context |

#### Recap

- A **window** of T inputs needs T+1 tokens and gives T predictions, one per position; the **context length** limits what the model can use and what it costs.
- **Targets are inputs shifted by one.** Every off-by-one variant trains silently; a test on distinct IDs catches them all.
- **Stride** controls overlap; the default (stride equal to context length) trains every next-token step once.
- Separate examples are **padded**, with an **attention mask** for the model and **ignored targets** for the loss; ignored padding provably does not change the loss.
- **Packing** joins documents with a **separator token** into full windows, eliminating padding at the cost of cross-document context, which can be masked using document IDs.
- **Bucketing** similar lengths cuts padding when examples must stay separate.
- `Dataset` indexes examples; `DataLoader` batches and shuffles them reproducibly with a seeded generator.
- Per-position training flattens `(batch, seq, vocab)` logits and `(batch, seq)` targets for cross-entropy.

#### Concept checks

1. A window has context length 8. How many tokens does it need, and how many predictions does it train?
2. Given the stream `[10, 11, 12, 13, 14, 15, 16]` and context length 3 with the default stride, list the inputs and targets of every window.
3. Why does a no-shift bug produce a near-zero loss?
4. What does a stride smaller than the context length gain and cost?
5. Why does doubling the batch size and halving the context length keep tokens per batch the same but not the learning task?
6. What are the three tensors `pad_batch` returns, and which one does the loss use to skip padding?
7. Why must padded targets be -100 rather than the pad ID?
8. When would you prefer left padding?
9. What does packing gain, and what does it risk?
10. Why is the separator inserted as an ID rather than by encoding the text `<|endoftext|>`?
11. What does `reshape(-1, vocab_size)` do to logits of shape `(4, 32, 2048)`?
12. Why is the bigram model's untrained loss higher than the even-spread reference?

#### Exercises

**Exercise 1 (count windows).** For a stream of 1,000 tokens, compute how many windows `TokenWindowDataset` gives for context lengths 32, 128, and 999, with the default stride and with stride 1. Check with code. How many times is each token predicted with stride 1?

**Exercise 2 (catch the bug).** Write three broken versions of `__getitem__` (no shift, wrong direction, shift by two) and confirm that `test_targets_are_inputs_shifted_by_one` fails for each. Then write a second test, on a decoded harbor sentence, that would fail for each in a way a human reading the failure message would understand immediately.

**Exercise 3 (padding and the loss).** Using the bigram model, train once on packed windows and once on padded sentences (with `pad_batch`, flattening and ignoring padding), for the same number of *real tokens*. Compare validation losses and training time. Then repeat the padded run with padded targets set to the pad ID, and describe the difference in the model's predictions after a full stop.

**Exercise 4 (bucketing).** Write `bucketed_batches(documents, batch_size, seed)` that sorts by length, cuts batches, and shuffles batch order, and measure padding waste against random batching for batch sizes 8, 32, and 128.

**Exercise 5 (document masks, preview).** For the first packed window from `packing.py`, build a `(32, 32)` boolean tensor that is `True` where position i may use position j: same document and j not after i. Print it as a grid of `#` and `.`. (Chapter 13.7 uses exactly this tensor.)

**Exercise 6 (a streaming dataset).** Write a dataset that reads token IDs from a file in chunks and yields windows without loading the whole file into memory, using a generator (Chapter 2.7) and PyTorch's `IterableDataset`. Decide how shuffling should work when you cannot index randomly, and justify your choice.

#### Suggested answers and acceptance criteria

**Concept checks**

1. 9 tokens; 8 predictions.
2. Window 0: inputs `[10, 11, 12]`, targets `[11, 12, 13]`. Window 1: inputs `[13, 14, 15]`, targets `[14, 15, 16]`. Two windows (a third would need tokens beyond 16).
3. Each target is identical to the input at the same position, so the model can copy its input; the loss goes to near zero without learning anything about language.
4. It gains more windows (and each token predicted with different amounts of context); it costs repetition of the same material, more compute per epoch, and higher overfitting risk.
5. The number of predictions per step is unchanged, but each prediction can draw on only half as much context, so long-range patterns can no longer be learned.
6. `input_ids`, `targets`, `attention_mask`. The loss uses `targets` containing -100 (its ignore index); the mask is for the model.
7. Cross-entropy skips -100; a pad ID would be a real target, training the model to predict padding (and, if the pad ID is also a real token, to predict that token wrongly).
8. In batched generation, so that every row's last real token is in the same column and the next token can be predicted for all rows at once.
9. It gains full windows with no wasted positions. It risks the model seeing unrelated earlier documents within a window.
10. So that text from users or documents containing those characters cannot produce the control token (Chapter 9.6).
11. It merges the first two axes, giving `(128, 2048)`: one row per (sequence, position) prediction.
12. The random starting logits differ from each other, so the model starts unevenly wrong, which cross-entropy penalizes more than an even spread.

**Exercise 1.** Default stride: 31 windows (context 32), 7 (128), 1 (999). Stride 1: 968, 872, and 1. With stride 1, most tokens are predicted up to T times (once from each window that includes them as a target). Acceptance: your numbers match `len(TokenWindowDataset(...))` and you can derive them from "a window needs T+1 tokens".

**Exercise 2.** Acceptance: each broken version fails the distinct-ID test. A good human-readable test decodes the first window of `"The keeper lit the lamp."` and asserts that the target text at position 0 is `" keeper"`, so the failure message shows which token was used instead.

**Exercise 3.** Acceptance: the two correct runs reach similar validation losses (they learn from the same real tokens), with padded batches doing more computation per real token; the run with pad IDs as targets shows a higher loss and, after `.`, predicts the pad token much more often. Report the numbers.

**Exercise 4.** Solution: [`code/solutions/ch11_bucketing.py`](../../code/solutions/ch11_bucketing.py); observed output in section 11.7; tested by `test_bucketing_reduces_padding`. Acceptance: bucketed waste is far below random waste at every batch size, and grows with batch size (larger batches span a wider range of lengths).

**Exercise 5.** Acceptance: the grid is a staircase of lower-left triangles, one per document, with `.` everywhere a position would look into a different document or into the future. Hint: the "same document" part compares `document_ids[i]` with `document_ids[j]` using broadcasting (Chapter 3.7); the "not after" part is `j <= i`.

**Exercise 6.** Acceptance: windows yielded from a file larger than you load at once; a test that the windows match `TokenWindowDataset` on a small file read whole. A sound shuffling design: keep a buffer of a few thousand windows and yield them in random order (a *shuffle buffer*), seeded, or shuffle the order of file chunks per epoch. Your write-up should say which randomness you gave up and why it is acceptable.

#### Checkpoint: what you can now do independently

You have finished Part 2. You can now:

- Take any text corpus to batches of fixed-shape inputs and targets, with a tested shift by one.
- Choose context length, stride, and batch size with their costs in mind.
- Batch variable-length examples with padding, masks, and ignored targets, and prove padding does not affect the loss.
- Pack documents with separators, measure the padding saved, and know how to mask document boundaries.
- Use `Dataset` and `DataLoader` reproducibly, and compute a loss over every position of a batch.

**Next:** Part 3 builds the transformer. Chapter 12 tours the main transformer architectures and explains why this book builds a decoder-only model.
