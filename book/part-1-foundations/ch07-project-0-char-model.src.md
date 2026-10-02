## Chapter 7: Project 0: Your First Trained Model, a Next-Character Predictor

[Back to index](../../README.md) · Previous: [Chapter 6](ch06-how-training-works.md) · Next: [Chapter 8](../part-2-text-to-inputs/ch08-text-unicode-bytes-tokens.md)

Everything in Part 1 comes together here. You will train a neural network on harbor text to predict the next character, using the honest data splits from Chapter 4, the network pieces from Chapter 5, and the training loop from Chapter 6. Then you will put it head to head with a counting model like Chapter 1's, on exactly the same data, and measure where each one succeeds and fails.

This is the book's warm-up project (Project 0). It is deliberately small, so it trains in seconds on a CPU, but it has every part of a real language-model project: a vocabulary, training examples built from raw text, a model, a training loop with validation and baselines, checkpoints, text generation, and an honest analysis of what the model does and does not do. Part 3 replaces the model with a transformer; the surrounding structure stays the same.

#### Learning outcomes

By the end of this chapter you will be able to:

1. Build a character vocabulary and turn raw text into (context, next character) training examples.
2. Explain one-hot inputs, why they are a reasonable first input format, and what they cost.
3. Train a next-character network with validation, baselines, and a run record, and read its results.
4. Compare a neural model with a counting model on seen and unseen contexts, and explain the difference.
5. Generate text from a trained model, and measure how much of it is well-formed, true, false, or copied.
6. Save a trained model as a checkpoint and reload it with identical behavior.
7. Use "overfit a single batch" as a debugging technique, and recognize the off-by-one target bug.

#### Prerequisites

- [Chapter 1](ch01-what-a-language-model-predicts.md): next-token prediction, generation loop, memorization, fluent falsehoods.
- [Chapter 3](ch03-tensors.md): shapes, int64 IDs, indexing with tensors, `flatten`.
- [Chapter 4](ch04-data-experiments-reproducibility.md): deduplication, `hash_split`, the synthetic harbor corpus, run records.
- [Chapter 5](ch05-neural-networks.md): `nn.Linear`, ReLU, logits, softmax, sampling with `torch.multinomial`.
- [Chapter 6](ch06-how-training-works.md): `fit`, `evaluate`, `train_step`, cross-entropy reference values, AdamW.

#### New terms in this chapter

| Term | Plain-English meaning | Section |
|---|---|---|
| Character vocabulary | The fixed list of characters the model knows, each with an integer ID | 7.2 |
| Context window (of a model) | The fixed number of previous characters the model receives | 7.3 |
| One-hot vector | A list of zeros with a single 1 at the position of one item's ID | 7.4 |
| `state_dict` | A dictionary of a module's parameter tensors, by name: what a checkpoint stores | 7.8 |
| Overfitting a single batch | Training on one small batch until the loss is near zero, to prove the pipeline can learn | 7.9 |
| Off-by-one target bug | Targets shifted one position early, so the answer is already in the input | 7.9 |

---

### 7.1 The problem: replace the count table with something learned

Chapter 1's counting model had two fatal weaknesses: it could say nothing about a context it had never seen, and it had no notion that different contexts might be similar. Chapters 5 and 6 introduced a model that computes scores for any input, and a training process that tunes it to data. This project tests whether that combination actually solves the counting model's problems on real (if simple) text.

We work with **characters** rather than words. Characters keep the vocabulary tiny (under 30 symbols), so the model and its inputs are small and training takes seconds. It also means that no word is ever "unknown": any word can be spelled. The cost is that a fixed number of characters covers much less text than the same number of words. Chapter 8 discusses this trade-off, and Chapter 9 resolves it with subword tokens.

#### Project definition

| | |
|---|---|
| **Problem** | Predict the next character of harbor text, and generate new text |
| **Data** | The synthetic harbor corpus (Chapter 4.2), deduplicated and split by sentence hash |
| **Success criteria** | (1) Validation loss far below the even-spread reference, and accuracy far above the "most common character" baseline; (2) the network matches the counting model on contexts counting has seen, and is clearly right on contexts it has not; (3) a single batch can be overfit to near-zero loss; (4) a saved checkpoint reloads with identical outputs; (5) a written analysis of generated text |
| **Prerequisite chapters** | 1–6 |
| **Hardware** | CPU; about 10 seconds per training run on the test machine. No GPU needed (`--device cuda` works if available; not tested by the author) |
| **Dependencies** | The Chapter 2 environment; nothing new |

---

### 7.2 A character vocabulary for the harbor text

A model works with integer IDs, not characters. The **vocabulary** is the fixed list of characters the model knows; a character's position in the list is its ID.

The module for this chapter:

File: [`code/llmfp/char_model.py`](../../code/llmfp/char_model.py)

```python
@@FILE code/llmfp/char_model.py@@
```

`CharVocabulary` decisions:

- **Built from the training text only.** Building it from all data would be a mild form of leakage (Chapter 4.4), and in real projects, validation text can contain characters the training text lacks. `encode` raises a clear `ValueError` naming the unknown character instead of a bare `KeyError`. Chapter 9 shows how byte-level tokenizers make unknown characters impossible.
- **Sorted.** `sorted(set(text))` gives the same IDs for the same text on every run and every machine. `list(set(text))` would not (Chapter 4.6).
- **Saved as JSON.** A model's output layer has one row per vocabulary entry, in vocabulary order. A model loaded with a different vocabulary would produce scores for the wrong characters, with no error. The vocabulary is therefore saved with every checkpoint (7.8).

On the harbor corpus, the vocabulary has 29 characters, as the training run in 7.5 prints: newline, space, period, the capital letters that begin sentences, and the lowercase letters used.

---

### 7.3 Turning text into (context, next character) examples

Each training example is a window of `context_size` characters (the input) and the character that follows it (the target). This is Chapter 1's sliding window, applied to characters. The fixed number of characters the model receives is its **context window**.

`make_examples` builds every window at once. `unfold(0, context_size, 1)` is a PyTorch operation that produces every run of `context_size` consecutive elements, stepping by 1: windows starting at position 0, 1, 2, and so on. The targets are the sequence from position `context_size` onward. Dropping the last window (`[:-1]`) keeps windows and targets the same length, because the final window has no next character. The test pins it down on a tiny case:

```text
ids:       [10, 11, 12, 13, 14]     context_size 3
contexts:  [10, 11, 12]  [11, 12, 13]
targets:        13            14
```

The whole training split is joined into one text with a newline after each sentence, so windows run across sentence boundaries. The model therefore also learns that a newline is followed by a capital letter, and what tends to come after a period. Validation text is prepared the same way, from validation sentences only.

---

### 7.4 One-hot inputs: the simplest way to feed characters to a network

A network multiplies its inputs by weights (Chapter 5.2). If we fed it raw IDs, it would treat character 20 as "twice as much" as character 10, which is meaningless. IDs are labels, not measurements. Chapter 10 makes this point central.

A **one-hot vector** avoids the problem. For a vocabulary of 29 characters, each character becomes a list of 29 numbers: all zeros except a single 1 at the character's ID. Every character is then equally different from every other, and no ordering is implied.

```python
import torch
from torch.nn import functional as F
F.one_hot(torch.tensor([2, 0]), num_classes=4)
# tensor([[0, 0, 1, 0],
#         [1, 0, 0, 0]])
```

(Illustrative; the same function runs inside `CharMLP.forward`, where the tests exercise it.)

`CharMLP.forward` one-hot encodes the whole `(batch, context_size)` tensor of IDs into `(batch, context_size, vocab_size)`, then `flatten(start_dim=1)` joins the last two axes into one long input list per example: 12 positions times 29 characters, so 348 numbers, of which exactly 12 are 1. The first linear layer then has one weight for every (position, character) pair, for every hidden unit. In effect, each hidden unit learns how much "character X at position P" counts toward its output.

What one-hot inputs cost:

- **Size.** The input grows with vocabulary size times context size. For a word-level or subword vocabulary of 50,000 tokens and a 1,000-token context, each input would have 50 million numbers. That is unworkable.
- **No sharing between similar items.** The weights for "a" and "e" are separate; nothing in the input says they are both vowels. The network can learn similarities in its hidden layer, but only from data about each character separately.
- **Position-specific weights.** "keeper" at positions 1–6 and at positions 4–9 use entirely different weights. The network must learn each pattern at each position.

Chapter 10 replaces one-hot inputs with learned *embeddings*, which fix the first two problems; Part 3's attention addresses the third. One-hot inputs are used here because they hide nothing: you can see exactly what the network receives.

---

### 7.5 The model and its training loop, with validation

The model, `CharMLP`, is Chapter 5's two-layer network with one-hot inputs, and training uses Chapter 6's `fit` without changes. The script ties everything together:

File: [`code/configs/char-model-cpu.toml`](../../code/configs/char-model-cpu.toml)

```toml
@@FILE code/configs/char-model-cpu.toml@@
```

File: [`code/scripts/ch07_train_char_model.py`](../../code/scripts/ch07_train_char_model.py)

```python
@@FILE code/scripts/ch07_train_char_model.py@@
```

```bash
python -m scripts.ch07_train_char_model
```

Observed output (under 10 seconds on the test machine):

```text
@@RUN python -m scripts.ch07_train_char_model@@
```

Reading the results against the success criteria:

- **Reference points.** Even spread over 29 characters gives a loss of 3.367 (Chapter 6.2's reference). Always guessing the most common character, space, gets 17% right. The trained network's validation loss of about 0.19 is far below the reference, and its accuracy above 90% is far above the baseline. **Criterion 1 is met.**
- **Training and validation track each other.** Validation loss stays close to training loss and keeps falling slowly, so the network is not overfitting at this size and number of epochs.
- **Why not 100%?** The corpus was generated by random choices: after "The keeper ", the next action is genuinely unpredictable. No model can do better than guess on those characters (Chapter 4.5's irreducible uncertainty). Accuracy near 92% means the network gets nearly everything right *except* the genuinely random choices.
- **The counting model is a strong baseline here.** With 12 characters of context, it saw 98.2% of validation contexts in training and gets about 90% right. On this very regular corpus, memorized 12-character windows go a long way. That is worth knowing: a fancier model is not automatically better, and you only find out by measuring against a simple baseline.

---

### 7.6 Comparing with the counting model on unseen contexts

The decisive comparison is in the block "Validation accuracy by whether the counting model had seen the context". It splits validation examples into two groups:

- **Seen contexts.** Both models do about equally well (about 91.5%). On familiar ground, a lookup table and a trained network learn the same thing from the same data.
- **Unseen contexts.** The counting model cannot answer at all: 0%. The network gets about 97% of them right. **Criterion 2 is met.** It has never seen these exact 12 characters, but it computes a sensible answer from what it learned about characters at each position.

With only about 100 unseen contexts at this size, the effect is clear but small. The solution to Exercise 1 sweeps the context size, which changes how many contexts are unseen:

```bash
python -m solutions.ch07_context_sweep
```

Observed output:

```text
@@RUN python -m solutions.ch07_context_sweep@@
```

Read the columns together. With short contexts (2 and 4 characters), almost every context was seen, and the two models are equal; neither can do better than about 70% and 86%, because a few characters are not enough information. As contexts grow, the counting model's coverage falls (83% at 24 characters) and its accuracy falls with it, from a peak near 90% to 76%. The network keeps improving (92% at 24 characters) because it still produces answers for unseen contexts, and gets most of them right. This is Chapter 1.11's prediction, measured: **the more specific the context, the more a lookup table suffers and the more generalization matters.**

---

### 7.7 Sampling text from the trained network

`sample_text` is the generation loop from Chapter 1.7: take the last `context_size` characters, get scores for every next character, choose one (greedily, or by sampling from the softmax shares, Chapter 5.7), append it, repeat. A prompt shorter than the context is padded on the left with newlines, the character that precedes every sentence in training text, playing the role Chapter 1's `<start>` marker played.

Look again at the samples at the end of the training run's output. The greedy line and most samples read like the corpus. Some are recombinations the corpus never contains. And some samples break down into strings that are not words at all, when one random choice of an unlikely character leads the model into a context unlike anything it saw.

How much of the generated text is right? The corpus was produced from a fixed table of which actor performs which actions ([`ch04_make_harbor_corpus.py`](../../code/scripts/ch04_make_harbor_corpus.py)). That table is a complete description of the harbor's "world", so we can check generated sentences against it. The Exercise 5 solution generates 200 sentences and sorts them:

```bash
python -m solutions.ch07_fact_check
```

Observed output:

```text
@@RUN python -m solutions.ch07_fact_check@@
```

Three findings:

- **Memorization.** A substantial share of generated sentences are exact copies of training sentences, the behavior Chapter 1.11 measured for counting models, appearing in a neural network.
- **Fluent but false.** A meaningful number of sentences are perfectly well-formed and claim things the world never contains, like boats selling fish. The network learned that "sold the fish" can follow "and" and that sentences end with a time phrase, but it has no representation of *who* can do *what* beyond the characters it can see. This is the same failure as Chapter 1's "the gulls lit the lamp at dusk", from a completely different kind of model. Its name, at LLM scale, is hallucination (Chapter 38.3).
- **Lost structure.** Many sentences are not well-formed: `"Before the storm."` starts with a time phrase and then ends early, because by the time "the storm" is complete, the 12-character window no longer contains "Before", and "the storm." is a very common ending. **The model cannot remember anything older than its window.** This is the limitation that motivates attention over long contexts (Part 3).

> **About these numbers.** The classifier is strict: a sentence must match the corpus's exact templates to count as well-formed. Exact counts depend on the seed and on training; your run with the same code, data, and seed should match the book's on the test machine's software versions. What should not change is the shape of the result: a mix of copies, true recombinations, false recombinations, and structural failures.

---

### 7.8 Saving and loading with `state_dict`

A trained model's parameters must be saved to be useful later. PyTorch's standard way is the **`state_dict`**: a dictionary mapping each parameter's name (`hidden.weight`, `hidden.bias`, `output.weight`, `output.bias`) to its tensor. `torch.save(model.state_dict(), path)` writes it; `model.load_state_dict(torch.load(path))` copies saved tensors into an existing model with matching names and shapes.

A `state_dict` holds only numbers. Recall Chapter 1.4: a checkpoint without the matching architecture is useless. `save_checkpoint` therefore writes three files:

| File | Contents | Why it is needed |
|---|---|---|
| `model.pt` | The `state_dict` | The learned parameters |
| `model_config.json` | `vocab_size`, `context_size`, `hidden` | To build a model of the right shape before loading parameters into it |
| `vocab.json` | The character list | To map IDs back to the right characters |

`load_checkpoint` follows the same order: build `CharMLP` from the configuration, then load the parameters into it. `load_state_dict` checks that every name and shape matches and raises an error otherwise, which is exactly what you want when a configuration and a weights file do not belong together. Chapter 22 meets that check when loading GPT-2's published weights into our own model.

Two safety notes:

- **`weights_only=True`.** A file saved with `torch.save` can, in general, contain arbitrary Python objects, and loading such a file can run code. `weights_only=True` restricts loading to tensors and simple containers, so loading a checkpoint you did not create cannot execute anything. Use it whenever you load files from elsewhere. Chapter 22.3 introduces *safetensors*, a format designed to hold only tensors.
- **`map_location`** loads tensors onto the device you ask for, so a checkpoint saved on a GPU machine loads on a CPU-only one.

The training script saved the checkpoint inside the run directory, reloaded it, and confirmed identical outputs on 100 validation examples. All samples were generated from the *reloaded* model. **Criterion 4 is met.** This checkpoint is enough to *use* the model. Resuming *training* exactly where it stopped also needs the optimizer's internal state (AdamW's running averages, Chapter 6.6) and the random number generator states; Chapter 19.9 builds full training checkpoints.

---

### 7.9 First debugging technique: overfit a single batch

When a training run fails to learn, the cause could be anywhere: the data, the targets, the model, the loss, the optimizer, the learning rate. The fastest way to narrow it down is to ask a much easier question: **can this setup learn even one small batch by heart?**

A model with tens of thousands of parameters should be able to memorize 32 examples almost perfectly. If it cannot, something in the pipeline is broken, and you have ruled out "the task is too hard" and "the model needs more training" in seconds.

```bash
python -m scripts.ch07_train_char_model --set overfit_one_batch=true
```

Observed output:

```text
@@RUN python -m scripts.ch07_train_char_model --set overfit_one_batch=true | tail -9@@
```

The loss falls toward zero and accuracy on the batch reaches 100%. **Criterion 3 is met.** When this check fails, typical causes are a missing `zero_grad` or `optimizer.step()` (Chapter 6.8), a learning rate far too small, a model whose output does not depend on its input, or a loss applied to the wrong tensors.

#### The check that passes for the wrong reason: the off-by-one target bug

Overfitting one batch proves the pipeline *can learn something*. It does not prove it is learning the *right* thing. The classic example is building targets one position too early, so each target is the last character of its own input window. The solution to Exercise 4 trains with exactly that bug:

```bash
python -m solutions.ch07_off_by_one
```

Observed output:

```text
@@RUN python -m solutions.ch07_off_by_one@@
```

Validation accuracy of 100% after two epochs, far better than the correct model. A beginner might celebrate. But the task the model learned is "repeat the last character you were shown", and greedy generation shows it: once the prompt ends in a space, the model repeats spaces forever. **A result that looks too good is a bug until proven otherwise** (Chapter 4.4). Three habits catch this bug:

1. **A test on a tiny, hand-checked case.** `test_make_examples_pairs_each_window_with_the_next_id` fixes the exact pairing on five IDs; `test_buggy_targets_are_the_last_character_of_each_window` shows what the bug looks like.
2. **Comparison with a baseline.** The counting model reached about 90%. A network that beats a strong baseline by a huge margin on a task with built-in randomness deserves suspicion.
3. **Looking at generated output**, not only at metrics.

Chapter 11.3 returns to this bug in its most common form, in transformer training data, and Chapter 20 builds a full debugging routine around these habits.

---

### 7.10 Project 0 review

#### Evaluation summary and failure cases

| Criterion | Evidence | Met? |
|---|---|---|
| Far below reference loss, far above baseline accuracy | Val loss ~0.19 vs 3.37; accuracy ~92% vs 17% | Yes |
| Matches counting on seen contexts; succeeds on unseen | ~91.5% vs ~91.7% seen; ~97% vs 0% unseen | Yes |
| Overfits one batch | Loss to ~0.0004, 100% on the batch | Yes |
| Checkpoint round trip | Identical outputs after reload | Yes |
| Analysis of generated text | Copies, true and false recombinations, structural failures counted | Yes |

Failure cases you have observed and should be able to explain:

- **Fluent but false** sentences: the model has no representation of facts beyond character patterns.
- **Malformed sentences** from limited context: nothing older than 12 characters influences the next prediction.
- **Garbled words after an unlucky sample**: one unlikely choice puts the model in an unfamiliar context, and errors compound. Chapter 21 shows how decoding settings reduce this.
- **Memorized sentences**: a share of output is copied verbatim from training data.

#### Debugging exercise

A colleague changed `make_examples` and reports a breakthrough: validation accuracy jumped from 92% to 100%, and the loss is almost zero. Before reading on, write down what you would check, in order. Then run the off-by-one solution and confirm the diagnosis. Acceptance: your list includes (1) comparing with the counting baseline and asking why the network beats it by so much, (2) generating text and inspecting it, and (3) checking the input/target pairing on a hand-made five-ID example with a test.

#### Reviewer checklist

Use this list when reviewing your own or someone else's version of this project:

- [ ] The vocabulary is built from training text only, sorted, and saved with the checkpoint.
- [ ] Data is deduplicated and split by sentence (or document), not by window; no validation sentence appears in training.
- [ ] A test pins the context/target pairing on a hand-checked example.
- [ ] Baselines (even-spread loss, most common character, counting model) are reported next to the model's results.
- [ ] Training and validation loss are both reported per epoch; evaluation uses `eval()` and `no_grad()`.
- [ ] The run has a record: configuration, environment, data fingerprint, metrics.
- [ ] Overfitting a single batch has been demonstrated.
- [ ] The checkpoint reload is verified to give identical outputs, and loading uses `weights_only=True`.
- [ ] Generated samples are shown, with a seed, and analyzed rather than only admired.
- [ ] Claims about generalization are backed by a comparison on unseen contexts.

#### Extensions that require your own decisions

1. **Larger context.** Train with `context_size` 32 or 48. Does accuracy keep improving? What happens to the parameter count, and why? Decide whether the extra cost is worth it, with numbers.
2. **A word-level comparison.** Adapt the model to predict next *words* using Chapter 1's word splitting. You will need to decide how to handle words that appear in validation but not training, and what vocabulary size is acceptable for one-hot inputs.
3. **Embeddings instead of one-hot** (preview of Chapter 10). Replace the one-hot step with `nn.Embedding(vocab_size, 16)` followed by flattening. Compare parameter counts and accuracy, and explain the difference.
4. **Fact-check-driven evaluation.** Turn the fact checker into an evaluation metric reported after every epoch. Decide whether you would select checkpoints by validation loss or by this metric, and argue for it.

---

### 7.11 Recap, concept checks, exercises, answers, and checkpoint

#### Recap

- A **character vocabulary**, built from training text and sorted, maps characters to IDs and must be saved with the model.
- `make_examples` turns text into (window, next character) pairs; the target is the character *after* the window.
- **One-hot** inputs represent each character as a list with a single 1. They imply no ordering, but they are large and share nothing between similar characters.
- The network reached over 90% validation accuracy, far above baselines, and matched the counting model on familiar contexts while answering unseen contexts correctly. The advantage grows with context length.
- Generated text mixes memorized copies, true recombinations, **fluent but false** recombinations, and structural failures from the limited window.
- A checkpoint is a `state_dict` plus the configuration and vocabulary needed to use it; load with `weights_only=True`.
- **Overfitting a single batch** quickly proves a pipeline can learn. A result that looks too good, such as the **off-by-one target bug**, needs tests, baselines, and generated samples to expose it.

#### Concept checks

1. Why is the vocabulary built from training text only, and why is it sorted?
2. For the IDs `[4, 8, 15, 16, 23, 42]` and a context size of 4, what are the contexts and targets?
3. Why can't the network be given raw character IDs as input numbers?
4. With a vocabulary of 29 characters and a context of 12, how many numbers are in one flattened one-hot input, and how many are nonzero?
5. On contexts the counting model had seen, the network and the counting model scored about the same. Why is that unsurprising?
6. Why does the counting model's accuracy fall as context grows, while the network's rises?
7. The model generated "Before the storm." Explain why, in terms of its context window.
8. What three files does a checkpoint contain here, and what goes wrong if the vocabulary file is lost or mismatched?
9. What does `weights_only=True` protect against?
10. A model cannot overfit a single batch of 32 examples. Name three possible causes.
11. Why does the off-by-one bug produce 100% validation accuracy?
12. Why is "the loss went down" not enough evidence that training worked?

#### Exercises

**Exercise 1 (context sweep).** Train the network and evaluate the counting model for context sizes 2, 4, 8, 16, and 24. Report counting accuracy and coverage, network accuracy, and network accuracy on contexts counting had not seen. Describe the trend in two sentences.

**Exercise 2 (what does it predict?).** Load your trained checkpoint and print the five highest-scoring next characters, with their softmax shares, for these prompts: `"The harbor m"`, `"The keeper l"`, `"lit the lamp"`, `"zzzzzzzzzzzz"` (if `z` is in the vocabulary; if not, explain the error). Explain each result from what is in the corpus.

**Exercise 3 (checkpoint test).** Write a test that trains a tiny `CharMLP` for a few steps on a short string, saves it, reloads it, and checks that outputs are identical, that the vocabulary is identical, and that loading into a model with a different `hidden` size fails with an error.

**Exercise 4 (the off-by-one bug).** Write a version of `make_examples` whose targets are one position too early, train with it, and report validation accuracy and a greedy sample. Then write a test that would have caught the bug.

**Exercise 5 (fact check).** Generate at least 200 sentences and classify each as well-formed and true, well-formed but false, or not well-formed, using the actor-action table in `ch04_make_harbor_corpus.py` as ground truth. Also count exact copies of training sentences. Summarize what the numbers say about the model.

**Exercise 6 (temperature preview).** In a copy of `sample_text`, divide the logits by a number before softmax (try 0.5, 1.0, and 2.0; Chapter 5.7 showed what scaling logits does to the shares). Generate samples with each and describe how the text changes. Run the fact checker on each setting if you did Exercise 5.

#### Suggested answers and acceptance criteria

**Concept checks**

1. Using validation text would leak information about it (and in real data, validation may contain characters training lacks). Sorting makes IDs identical on every run, unlike set order.
2. Contexts `[4, 8, 15, 16]` and `[8, 15, 16, 23]`; targets `23` and `42`.
3. The network multiplies inputs by weights, so it would treat a larger ID as "more" of something. IDs are labels, not measurements.
4. 12 positions times 29 characters gives 348 numbers, of which 12 are 1.
5. Both learn which character tends to follow each familiar context from the same data; on familiar contexts the lookup table is already a good answer.
6. Longer contexts are more often unseen, so the counting model has no answer more often; the network still computes an answer and benefits from the extra information in the longer context.
7. When "the storm" ends, the 12-character window no longer includes "Before", so the model sees only a context that very often ends a sentence, and predicts a period.
8. `model.pt`, `model_config.json`, `vocab.json`. A missing or mismatched vocabulary maps the output scores to the wrong characters; the model runs but produces nonsense, with no error.
9. Executing arbitrary code hidden in a malicious or untrusted checkpoint file.
10. Any three of: missing `zero_grad` or `optimizer.step()`; learning rate far too small; loss computed on the wrong tensors or with softmax applied first; the model ignoring its input (for example, a broken forward pass); targets that do not correspond to inputs.
11. The target is already present in the input (the last character of the window), so the model only has to copy it, which is trivial to learn.
12. A loss can fall for wrong reasons (leakage, the off-by-one bug, a trivial task). Compare with baselines, look at generated output, and test the data pipeline on hand-checked cases.

**Exercise 1.** Solution: [`code/solutions/ch07_context_sweep.py`](../../code/solutions/ch07_context_sweep.py); observed output in section 7.6. Acceptance: your table shows counting coverage falling and network accuracy rising with context size, with the two models roughly equal at short contexts, and your two sentences say why.

**Exercise 2.** The author's checkpoint gave: after `"The harbor m"`, `a` with a share of 0.993 (only "master" follows "harbor m" in the corpus); after `"The keeper l"`, `i` with 0.987 ("lit"); after `"lit the lamp"`, a space (0.844) or a period (0.155), because "lit the lamp" is followed either by a time phrase or "and" (space) or, in sentences that start with a time phrase, by the end of the sentence (period). The harbor corpus contains no `z`, so `"zzzzzzzzzzzz"` raises `ValueError: character 'z' is not in the vocabulary`, which is the intended behavior. Acceptance: your top candidates match in kind (exact shares depend on training), and each explanation points to what the corpus contains.

**Exercise 3.** The test `test_checkpoint_round_trip_gives_identical_outputs` in [`code/tests/test_char_model.py`](../../code/tests/test_char_model.py) covers the first two parts. For the third, build `CharMLP(CharModelConfig(vocab_size, 4, 32))` and call `load_state_dict` with the saved `model.pt` contents; acceptance: `pytest.raises(RuntimeError)` passes, and you can point to the "size mismatch" message naming the parameter.

**Exercise 4.** Solution: [`code/solutions/ch07_off_by_one.py`](../../code/solutions/ch07_off_by_one.py); output and discussion in section 7.9; the test is `test_buggy_targets_are_the_last_character_of_each_window`. Acceptance: you report near-100% accuracy, a degenerate greedy sample, and a test on a hand-checked case that fails for the buggy version and passes for the correct one.

**Exercise 5.** Solution: [`code/solutions/ch07_fact_check.py`](../../code/solutions/ch07_fact_check.py), with classifier tests in `test_char_model.py`; observed output in section 7.7. Acceptance: counts for the three categories and copies, an example of each, and a summary that distinguishes memorization, false recombination, and structural failure, connecting the last to the context window.

**Exercise 6.** Acceptance: samples for each setting and a description. Expect dividing by 0.5 (gaps doubled) to produce text closer to the most common patterns, with fewer garbled words and more copies; 2.0 (gaps halved) to produce more varied text with many more misspellings and malformed sentences. Chapter 21.4 introduces this as the *temperature* setting.

#### Checkpoint: what you can now do independently

You have finished Part 1. You can now:

- Take raw text through vocabulary construction, honest splitting, and example creation, with tests that pin down the input/target pairing.
- Build, train, evaluate, save, reload, and sample from a small neural language model, with run records and baselines.
- Show, with measurements, why a trained network generalizes where a lookup table cannot, and where it still fails.
- Analyze generated text for memorization, fluent falsehoods, and structural failures, rather than judging by impression.
- Debug a training pipeline by overfitting a single batch, comparing with baselines, and inspecting outputs, and recognize results that are too good to be true.

**Next:** Part 2 makes the inputs serious. [Chapter 8](../part-2-text-to-inputs/ch08-text-unicode-bytes-tokens.md) explains text encoding, Unicode, and bytes, and why tokens are not words.
