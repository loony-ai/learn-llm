## Chapter 10: Embeddings and Position

[Back to index](../../README.md) · Previous: [Chapter 9](ch09-byte-pair-encoding.md) · Next: [Chapter 11](ch11-sequences-batches-targets.md)

A tokenizer turns text into token IDs. A network needs numbers it can compute with. Chapter 7 bridged the gap with one-hot vectors, and section 7.4 listed their costs: they grow with the vocabulary, they share nothing between related tokens, and they are expensive. Every modern language model uses a different bridge, the **embedding table**: a learned list of numbers for each token. This chapter builds embeddings, trains them, looks at what they learn (honestly, including where they learn less than you might hope), and then addresses a problem that embeddings alone do not solve: **word order**.

#### Learning outcomes

By the end of this chapter you will be able to:

1. Explain why token IDs cannot be used as numeric inputs, and what one-hot vectors cost at realistic vocabulary sizes.
2. Describe an embedding table as a learned lookup, and show it is equivalent to one-hot times a table.
3. Explain vectors, and what cosine similarity reports, from observed behavior.
4. Train models with embeddings and inspect what the learned tables encode, without over-interpreting them.
5. Show, with a measurement, what a model loses when it cannot see word order.
6. Add learned position embeddings, use them correctly, and state their limitations.

#### Prerequisites

- [Chapter 3.5](../part-1-foundations/ch03-tensors.md): indexing a table with a tensor of IDs (`table[ids]`).
- [Chapter 5](../part-1-foundations/ch05-neural-networks.md): linear layers, activations, output layer.
- [Chapter 7.4](../part-1-foundations/ch07-project-0-char-model.md): one-hot inputs and their costs.
- [Chapter 9](ch09-byte-pair-encoding.md): the Project 1 BPE tokenizer, saved at `data/tokenizer/harbor-bpe-2048.json`.

#### New terms in this chapter

| Term | Plain-English meaning | Section |
|---|---|---|
| Vector | A list of numbers treated as a single item, such as one row of a table | 10.3 |
| Embedding table | A learned table with one row (vector) per token ID | 10.3 |
| Embedding (of a token) | The row of the embedding table for that token | 10.3 |
| Embedding dimension (`d_model`) | How many numbers each embedding has | 10.3 |
| Cosine similarity | A score of how closely two vectors point in the same direction: 1 same, 0 unrelated, -1 opposite | 10.4 |
| Bag of tokens | Treating a context as an unordered collection, losing word order | 10.6 |
| Positional embedding | A learned vector for each position, added to the token embedding so position is visible | 10.7 |
| Context length (of positions) | The number of positions a model has embeddings for; the longest sequence it can take | 10.7 |

---

### 10.1 The problem: token IDs are labels, not measurements

File: [`code/examples/ch10/ids_are_labels.py`](../../code/examples/ch10/ids_are_labels.py)

```python
@@FILE code/examples/ch10/ids_are_labels.py@@
```

Observed output:

```text
@@RUN python examples/ch10/ids_are_labels.py@@
```

The IDs of `' dawn'`, `' dusk'`, `' noon'`, and `' night'` happen to be close together, because the Project 1 tokenizer learned those merges at a similar point in training (Chapter 9.4): they are similarly frequent in the corpus. That is an accident of the merge order, not a property of the words. The difference between two IDs, or the token "halfway" between them, means nothing. Retrain the tokenizer on slightly different text and every ID changes.

A network multiplies inputs by weights (Chapter 5.2). Fed raw IDs, it would treat token 596 as "more" than token 422. So token IDs must be used as *labels*: to look something up, never as quantities. Chapter 7 looked up a one-hot vector. This chapter looks up something much better.

---

### 10.2 One-hot inputs revisited, and their cost

A one-hot vector for a vocabulary of size V is V numbers with a single 1. The first linear layer then has one weight per (input position, vocabulary entry, hidden unit). With the Project 1 tokenizer's 2,048 tokens, a context of 8 tokens, and 128 hidden units, the milestone script in section 10.5 reports what that layer would need: **2,097,152 weights**, before learning anything else. With a 50,000-token vocabulary and a 1,000-token context, the same design would need billions of weights in the first layer alone.

There is a second, subtler cost. Every vocabulary entry has its own separate weights, and every one-hot vector is equally different from every other. Nothing lets what the model learns about `' dawn'` help with `' dusk'`. Each token must be learned from its own examples.

---

### 10.3 Embedding tables: a learned lookup

An **embedding table** gives every token ID one row of numbers, `d_model` numbers long (the **embedding dimension**). A list of numbers treated as one item like this is called a **vector**, and a token's row is its **embedding**. Using the table is a lookup: the IDs select rows. The numbers in the rows are parameters, set randomly at first and adjusted by training like any others.

File: [`code/examples/ch10/embedding_lookup.py`](../../code/examples/ch10/embedding_lookup.py)

```python
@@FILE code/examples/ch10/embedding_lookup.py@@
```

Observed output:

```text
@@RUN python examples/ch10/embedding_lookup.py@@
```

What to take from it:

- **Shapes.** `nn.Embedding(6, 3)` has a weight of shape `(6, 3)`: one row per token. IDs of shape `(batch, seq)` become vectors of shape `(batch, seq, 3)`. Every token position now carries a vector, which is the `(batch, sequence, features)` layout Chapter 3.3 said transformers use.
- **Equivalence with one-hot.** One-hot vectors multiplied by the table give exactly the same result. An embedding layer is a linear layer applied to one-hot inputs, implemented as a lookup so the one-hot vectors never have to exist. That is why it is cheap: the cost depends on `d_model`, not on the vocabulary size.
- **Only looked-up rows learn.** The backward pass gave gradients only to rows 1 and 4, the IDs that were used. A token that never appears in the training data keeps its random starting row forever. Section 10.5 has to account for this when inspecting embeddings, and Part 4 discusses why a tokenizer trained on different text than the model can leave many such untrained rows.
- **IDs must fit the table.** An ID equal to or beyond the number of rows is an `IndexError`. When you see this error in model code, the first suspect is a tokenizer whose vocabulary is larger than the model's embedding table (Chapter 8.6).

The book's embedding module, which Part 3's transformer will use:

File: [`code/llmfp/model/embeddings.py`](../../code/llmfp/model/embeddings.py)

```python
@@FILE code/llmfp/model/embeddings.py@@
```

`TokenAndPositionEmbedding` is explained in section 10.7, and the two similarity helpers in section 10.4. The package's `__init__.py` lists what it holds:

File: [`code/llmfp/model/__init__.py`](../../code/llmfp/model/__init__.py)

```python
@@FILE code/llmfp/model/__init__.py@@
```

---

### 10.4 Vectors as lists of learned features; similarity and cosine similarity

Why would a list of 32 numbers per token help a model more than an ID? Because training adjusts each token's numbers to whatever makes the model's predictions better. Tokens that call for similar predictions tend to be pushed toward similar numbers, and then everything the model computes from those numbers treats them similarly. In principle, what the model learns from `' dawn'` can then carry over to `' dusk'`.

It is tempting to think of each of the 32 numbers as a "feature" with a meaning, such as "is a time of day" or "is plural". **That picture is mostly a teaching analogy.** Individual dimensions of learned embeddings rarely correspond to a single human concept; whatever structure exists is usually spread across many dimensions at once. Research on interpreting learned representations is active, and its findings are partial (Chapter 1.12).

What we *can* measure reliably is whether two vectors point in a similar direction. **Cosine similarity** does that:

File: [`code/examples/ch10/cosine_similarity.py`](../../code/examples/ch10/cosine_similarity.py)

```python
@@FILE code/examples/ch10/cosine_similarity.py@@
```

Observed output:

```text
@@RUN python examples/ch10/cosine_similarity.py@@
```

Cosine similarity is 1 for vectors pointing the same way (even if one is ten times longer: length is ignored), close to 1 for slightly different directions, 0 for vectors at right angles, which in practice means "unrelated", and -1 for opposite directions. It is the standard way to compare embeddings, and you will use it again for search in Chapter 32. In `embeddings.py`, `cosine_similarity_matrix` first rescales every vector to length 1 (`F.normalize`), then compares all pairs with one matrix multiplication; `nearest_neighbors` finds the rows most similar to a given row.

---

### 10.5 Training embeddings and inspecting what they learn

The milestone model, `EmbeddingMLP`, is a next-token predictor over the Project 1 tokenizer's vocabulary. It looks up embeddings for the previous 8 tokens and combines them in one of three ways (`mode`), studied in sections 10.6 and 10.7:

File: [`code/llmfp/model/embedding_mlp.py`](../../code/llmfp/model/embedding_mlp.py)

```python
@@FILE code/llmfp/model/embedding_mlp.py@@
```

The script trains all three on harbor text, then examines the first model's learned tables:

File: [`code/configs/embedding-mlp-cpu.toml`](../../code/configs/embedding-mlp-cpu.toml)

```toml
@@FILE code/configs/embedding-mlp-cpu.toml@@
```

File: [`code/scripts/ch10_train_embedding_model.py`](../../code/scripts/ch10_train_embedding_model.py)

```python
@@FILE code/scripts/ch10_train_embedding_model.py@@
```

```bash
python -m scripts.ch10_train_embedding_model
```

Observed output (about 15 seconds on the test machine):

```text
@@RUN python -m scripts.ch10_train_embedding_model@@
```

First, the cost. The `concat` model has about 363,000 parameters in total, while a one-hot input layer for the same context would need about 2.1 million weights by itself. Most of the remaining parameters are in the output layer, whose 2,048 rows are mostly for tokens that never occur in harbor text: only 68 of the 2,048 vocabulary entries appear in training. That is a cost of using a tokenizer trained on broader text than the model's data.

Now the inspection, which is more interesting for what it does *not* show.

**The input embeddings show essentially no grouping by role.** Average cosine similarity among actors (keeper, fishers, children, gulls, boats, master), among times of day, and among verbs is about zero, the same as between groups. The nearest neighbors of `' keeper'` in the input table are a jumble (`' warned'`, `' fog'`, `' after'`). In this small model, trained on templated text, the input embeddings did not organize themselves by meaning.

**The output layer's rows group strongly.** The output layer has one row per candidate next token (Chapter 5.6). Its rows are also learned vectors, one per token, and they *are* organized: times of day average 0.70 similarity with each other, and the nearest neighbors of `' dawn'` are `' night'`, `' dusk'`, and `' noon'`. Those of `' keeper'` are the other actors, and those of `' lit'` are other things the keeper does (climbed, fixed, rang, cleaned).

Why the difference? The output layer scores every candidate next token from the same hidden values. Tokens that are likely in the same situations ("after `at`, a time of day comes next") need similar rows, so that they score high together; training pushes them together directly. Input embeddings, in the `concat` model, feed into position-specific weights in the hidden layer, so there is more than one way to make predictions work, and nothing forced a tidy arrangement.

Three lessons, all of which recur later:

1. **Learned representations can encode useful structure, and where it appears depends on the architecture and the data.** Large models trained on huge corpora do show strong structure in their input embeddings; this tiny model on templated text did not. Do not generalize from either case to the other.
2. **Inspect, measure, and report what you find, including null results.** A neighbor list from one well-chosen word can be made to look meaningful. The averaged group comparison is harder to fool yourself with.
3. **Input and output tables are both "one vector per token".** Chapter 16.5 shows a common design that uses one table for both (weight tying).

---

### 10.6 Why order matters: what a bag of tokens loses

Embeddings give each token a vector, but a context is a *sequence* of tokens. How should a model combine them?

The `concat` mode places the vectors side by side, so the first token's numbers always go to the same hidden-layer weights: order is preserved by position in the list, as with one-hot inputs. This has the position-specific cost from Chapter 7.4 and works only for a fixed number of tokens.

A tempting alternative is to combine the vectors in a way that does not depend on how many there are, such as averaging them. That treats the context as a **bag of tokens**: an unordered collection. The problem is that order carries meaning:

File: [`code/examples/ch10/bag_of_tokens.py`](../../code/examples/ch10/bag_of_tokens.py)

```python
@@FILE code/examples/ch10/bag_of_tokens.py@@
```

Observed output:

```text
@@RUN python examples/ch10/bag_of_tokens.py@@
```

"The keeper fed the gulls" and "The gulls fed the keeper" contain the same tokens. To a bag model they are identical. The milestone measured the cost on prediction: the `bag` model reached about 65% validation accuracy against about 71% for `concat`, and a noticeably higher loss. It cannot tell "the keeper" (followed by a keeper action) from "keeper the".

The `bag` mode in `EmbeddingMLP` passes each token's embedding through the hidden layer separately and then averages. Each token is processed identically wherever it sits, so the result cannot depend on order.

---

### 10.7 Learned positional embeddings

The fix used by GPT-2 and many transformers: make position part of each token's vector. A second embedding table has one row per **position** (0, 1, 2, ... up to the maximum sequence length). Each token's vector becomes its token embedding *plus* its position's embedding. The same token at two positions now has two different vectors, and anything computed from them can depend on position.

`TokenAndPositionEmbedding` (section 10.3's listing) does exactly this. `torch.arange(sequence_length)` produces the position IDs 0, 1, 2, ...; the position table turns them into vectors of shape `(seq, d_model)`; broadcasting (Chapter 3.7) adds them to every sequence in the batch. The `bag_position` mode uses it, and it changed the result substantially: about 72% validation accuracy, from 65% for the plain bag. Position information restored what the bag lost, and here even slightly beat `concat`.

#### The averaging trap

Position embeddings only help if the model does something with them *before* the information is averaged away. The last two lines of the bag example show it:

- "bag + positions, averaged directly": the two sentences still look identical.
- "bag + positions, ReLU layer first": they now differ.

Averaging the sums of token and position vectors gives the average of the token vectors plus the average of the position vectors, and the second part is the same for every context of that length. It is a constant; the order information cancels out. A nonlinear step applied to each token's vector *before* combining keeps it, because the activation (Chapter 5.4) responds to the token and the position together. The author's first draft of the milestone made exactly this mistake; Exercise 5 has you reproduce it, and a test (`test_averaging_first_cancels_position_information`) guards against it. In a transformer, the nonlinear, position-sensitive step is attention itself (Chapter 13), which weighs tokens differently depending on their vectors, including the position part.

#### Limitations of learned positions

- **A fixed maximum length.** There are only `context_length` rows. A longer sequence has no position vectors, and `TokenAndPositionEmbedding` raises an error rather than failing obscurely. A model's maximum context is often set this way, at training time.
- **Rarely seen positions are poorly trained.** If most training sequences are short, embeddings for late positions get few updates (only looked-up rows learn, section 10.3).
- **Absolute, not relative.** "Position 7" means the same in every sequence, but what usually matters in language is relative position: the word just before, three words back. A model with absolute positions has to learn relative patterns separately for every position.

---

### 10.8 Preview of other positional methods

Learned absolute positions are one of several designs, an area where practice has changed over time:

- **Fixed (sinusoidal) position vectors**, used in the original transformer: the position vectors are computed from a fixed recipe instead of being learned, so they exist for any position, though models still tend to perform poorly on sequences much longer than those seen in training.
- **Rotary positional embeddings (RoPE)**, used by many recent open models: instead of adding a position vector, they rotate parts of the vectors used inside attention by an amount that depends on position, so that attention scores depend on *relative* distance. Chapter 15.7 implements RoPE and shows its behavior in code.
- **Relative position biases** in attention, and several other variants, discussed at an overview level in Chapter 43.4 together with techniques for extending context length.

Which works best is an empirical question that depends on model size, training data, and the lengths you need. This book builds learned absolute positions first, because they are the simplest to understand, then RoPE.

---

### 10.9 Common mistakes, recap, concept checks, exercises, answers, and checkpoint

#### Common mistakes

| Symptom | Likely cause | Fix |
|---|---|---|
| `IndexError: index out of range in self` in an embedding | A token ID at or beyond the table size: tokenizer and model mismatched, or a padding/special ID not included | Check `tokenizer.vocab_size` against the embedding's number of rows |
| Error or nonsense for long inputs | Sequence longer than the position table | Truncate or window inputs (Chapter 11); know the model's context length |
| Model ignores word order | Combining token vectors with an order-insensitive operation, or adding positions and then averaging immediately | Concatenate, or apply a nonlinear (or attention) step to token-plus-position vectors before combining |
| Embedding inspection shows meaningful-looking neighbors for untrained tokens | Rows for tokens absent from training are still random | Restrict analysis to tokens seen in training |
| Strong claims from one nearest-neighbor list | Cherry-picking | Measure averaged similarities across groups and report null results |
| Raw IDs fed into a linear layer | Treating labels as quantities | Always look IDs up in an embedding table |

#### Recap

- Token IDs are labels; their numeric values carry no meaning.
- One-hot inputs are equivalent to an embedding lookup but cost a weight per vocabulary entry per input position; embedding tables cost `d_model` numbers per token and are looked up directly.
- Only rows that are looked up receive gradients; tokens absent from training keep random rows.
- Cosine similarity compares direction: 1 same, 0 unrelated, -1 opposite. Individual embedding dimensions rarely have a human meaning.
- In the milestone model, the output layer's rows grouped tokens by role strongly; the input embeddings did not. Where structure appears depends on the model and the data.
- A **bag of tokens** loses order and measurably worse predictions follow. **Positional embeddings** restore order, but only if a nonlinear step sees token and position together before combining.
- Learned absolute positions limit the maximum sequence length and treat positions absolutely; RoPE (Chapter 15.7) is a common alternative.

#### Concept checks

1. Why can't token IDs be fed to a linear layer as numbers?
2. What is the shape of `nn.Embedding(2048, 32)` applied to IDs of shape `(16, 8)`?
3. Explain why an embedding lookup gives the same result as one-hot vectors times the table, and why the lookup is cheaper.
4. After one training step on a batch containing tokens 5 and 9, which rows of the embedding table changed?
5. Two embedding vectors have cosine similarity 0.98. What does that tell you, and what does it not tell you?
6. In the milestone, why did the output layer's rows group times of day together?
7. Why were nearest neighbors computed only among tokens seen in training?
8. "The keeper fed the gulls" and "The gulls fed the keeper": which of the three modes can tell them apart?
9. Why does averaging token-plus-position vectors directly not preserve order?
10. A model has 256 position embeddings. What happens if you give it 300 tokens?
11. Why might "absolute" positions be less natural for language than relative ones?
12. What happens to a token's embedding if that token never appears in the training data?

#### Exercises

**Exercise 1 (shapes).** For `TokenAndPositionEmbedding(vocab_size=2048, context_length=64, d_model=48)`, predict: the number of parameters, the output shape for IDs of shape `(4, 64)`, and what happens for shape `(4, 65)`. Check with code.

**Exercise 2 (cosine by hand).** Without code, decide which pairs have cosine similarity 1, 0, or -1: `[1, 1]` and `[3, 3]`; `[1, 0]` and `[0, 5]`; `[2, -1]` and `[-4, 2]`; `[1, 2]` and `[2, 1]` (this one is none of them: is it closer to 1 or 0?). Check with `F.cosine_similarity`.

**Exercise 3 (embedding size).** Rerun the milestone with `d_model` 4, 16, 32, and 64 (`--set d_model=4`). Record validation accuracy and the role-similarity table for each. Does a larger embedding help accuracy here? Does it change where role structure appears?

**Exercise 4 (seen and unseen rows).** After training the `concat` model, compute the average size of the embedding rows (the length of each vector, `row.norm()`) for tokens that occurred in training and for those that did not. Explain the difference, and the role of weight decay in AdamW (Chapter 6.6) in it.

**Exercise 5 (the averaging trap).** Write a variant of the `bag_position` model that averages the token-plus-position vectors *before* the hidden layer. Train it with the same settings and compare its validation accuracy with `bag` and with the correct `bag_position`. Write a test showing it gives identical outputs for a context and its reverse.

**Exercise 6 (relative position, informally).** Train the `concat` model with `context_size` 2, 4, 8, and 16. Explain the trend in accuracy using what you know about the corpus, and say which positions in the context the model most needs for predicting the next token. (Optional: measure it by shuffling one position at a time in the validation contexts and recording the accuracy drop.)

#### Suggested answers and acceptance criteria

**Concept checks**

1. The layer multiplies inputs by weights, so it would treat a larger ID as "more" of something, while IDs are arbitrary labels.
2. `(16, 8, 32)`.
3. Multiplying a one-hot vector by the table picks out exactly the row at the 1; the lookup picks the same row directly, without building a vector of thousands of zeros or multiplying by them.
4. Only rows 5 and 9 (plus, through AdamW's weight decay, a tiny shrink of all rows if decay is applied to the table).
5. They point in almost the same direction, so the model will tend to treat them similarly in computations that depend on direction. It does not tell you the tokens mean the same thing, nor which dimensions are responsible.
6. Every candidate is scored from the same hidden values, and times of day are likely in the same situations (after "at"), so training pushes their rows toward similar directions so they score high together.
7. Rows for tokens never seen are never updated and stay random, so they would add meaningless neighbors.
8. `concat` and `bag_position`. `bag` cannot.
9. The average of sums is the sum of averages, and the average of the position vectors is the same for every context, a constant that carries no order information.
10. There is no position vector for positions 256 and beyond; the book's module raises an error (other code may fail differently or silently truncate).
11. Language patterns are mostly about relative distance ("the word before", "the subject a few words back"), which absolute positions must learn separately at every position.
12. It keeps its random initial values (shrunk slightly by weight decay, if applied), so it carries no learned information.

**Exercise 1.** Parameters: 2,048 rows of 48 plus 64 rows of 48, which `count_parameters` reports as 101,376. Output `(4, 64, 48)`; the 65-token input raises `ValueError` mentioning 64 positions. Acceptance: predictions checked in code.

**Exercise 2.** 1; 0; -1; the last is about 0.8, closer to 1. Acceptance: checked with code.

**Exercise 3.** Acceptance: a table of accuracy and role similarity per `d_model`. Expect accuracy to be similar from 16 upward (the harbor task is small), worse at 4; and the output-layer grouping to persist across sizes while the input-table grouping stays weak. Your conclusion should be stated for this data only.

**Exercise 4.** The author's run (10 epochs, default settings): rows of seen tokens grew on average from a length of 5.64 to 5.98, while rows of unseen tokens shrank from 5.61 to 5.48. Acceptance: unseen-token rows have a noticeably different average size from seen ones. Seen rows grew or changed under training; unseen rows were only shrunk by AdamW's default weight decay of 0.01 at every step, because weight decay applies to all parameters, used or not. You explain both effects and note that unseen rows carry no learned information either way.

**Exercise 5.** Solution: [`code/solutions/ch10_average_first.py`](../../code/solutions/ch10_average_first.py); the test is in [`code/tests/test_embeddings.py`](../../code/tests/test_embeddings.py). Observed output:

```text
@@RUN python -m solutions.ch10_average_first@@
```

Acceptance: the averaged-first variant scores close to the plain bag and well below the correct `bag_position`; its outputs for a context and its reverse are identical; you can explain why with the "average of sums" argument. (Its numbers are not exactly the bag's, because it is a slightly different model, with the hidden layer after the average rather than before. Neither can see order.)

**Exercise 6.** Acceptance: accuracy rises from context 2 to 8 and then flattens or changes little, because most of what predicts the next token in this corpus is within the current sentence's last few tokens (the actor determines the possible actions; "at"/"in the" predicts a time). If you did the optional measurement: shuffling the most recent positions costs the most accuracy, and shuffling the earliest positions very little. That is a measured case for why relative position matters more than absolute.

#### Checkpoint: what you can now do independently

You can now:

- Replace one-hot inputs with embedding tables and explain the equivalence and the savings.
- Measure similarity between learned vectors and inspect a trained model's tables without over-interpreting them.
- Demonstrate, with a measurement, what a model loses without word order.
- Add learned positional embeddings correctly, avoid the averaging trap, and state their limits.

**Next:** [Chapter 11](ch11-sequences-batches-targets.md) turns a stream of token IDs into training batches for a transformer: context windows, the input/target shift, padding, masks, packing, and PyTorch's `DataLoader`.
