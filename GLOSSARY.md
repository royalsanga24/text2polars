# Glossary

Plain-language definitions of the jargon. Terms we've actually hit in this
project are marked **[used]** with where they came up — those are the ones
you'll remember, because you have a story attached to them.

---

## 1. Measuring things

**Eval** **[used]** — a test suite for something with no single right answer.
Fixed inputs, a definition of "good", and a function that returns a number. The
number on its own means nothing; comparing two of them is the entire point.

**Benchmark** — an eval that's published so other people can compare against
it. Same machinery, different audience.

**Baseline** **[used]** — the simplest approach that produces a number. Usually
just a prompt. Anything more complicated has to beat it or it wasn't worth
building. *Ours: 35.4% from a few-shot prompt.*

**Ceiling** **[used]** — the best score anyone achieves, showing how much room
there is. *Ours: Claude at 100%, so the whole gap is closeable.*

**Saturated** **[used]** — everyone scores near the top, so the benchmark can't
tell good from great any more. *Ours is saturated at the frontier — fine as a
small-model diagnostic, weak as a published benchmark.*

**Stratify** **[used]** — report the score broken into categories instead of one
average, so a single dominant error can't hide everything else. *Ours: 0% on
`stale_api` was invisible in the overall 21%.*

**Calibrate** **[used]** — tune a tool's settings by testing it on cases where
you already know the right answer, instead of guessing the settings. *We
calibrated the contamination checker on deliberately planted cheats.*

**Noise** **[used]** — a difference that's just luck, not a real effect. With
small test sets, most small differences are noise. *Our v1-vs-v2 prompt
difference looked real and wasn't.*

**p-value** **[used]** — "if there were no real effect, how often would I see a
result this lopsided by pure chance?" Below 0.05 is the usual bar for "probably
real". *Our 3B-vs-1.5B comparison gave p = 1.0 — pure chance.*

**Arm** **[used]** — one version of an experiment you are comparing. From
medical trials ("treatment arm", "control arm"). Ours: no training / SFT only /
CPT only / CPT then SFT.

**Parity check** **[used]** — before comparing two things, confirm the setup is
genuinely identical. Caught a 12-point difference between two tools running
"the same" model (F27).

**Reference runtime** **[used]** — picking ONE tool to do all your measuring
with, so comparisons are fair. Ours is MLX at full precision.

**Statistical power** — whether your test is big enough to notice the size of
difference you care about. A 48-question test can spot a 20-point jump and is
blind to a 5-point one.

**Held-out / test set** **[used]** — data the model never trains on and you
rarely look at. Every time you tune against it, you burn a little of its ability
to tell you the truth.

**Dev set (or validation set)** **[used]** — the data you *do* look at
repeatedly while iterating. Kept separate from the test set for that reason.

**Overfitting** — when a model (or a person) learns the specific examples rather
than the general skill. Scores great on what it saw, badly on anything new.

**Leakage** **[used]** — information from the test reaching the training,
inflating results. The general category.

**Contamination** **[used]** — the most common kind of leakage: actual test
examples appearing in training data. The student saw the exam paper.

**Decontamination** **[used]** — actively finding and removing those. *Our
`curation/decontaminate.py`.*

**Error analysis** **[used]** — reading the actual failures instead of only
looking at the score. Consistently the highest information-per-minute activity
available. *It's how we found the pandas-interference story.*

**Ablation** — removing one piece to see how much it was contributing. "We
trained without CPT and scored 12 points lower" — that's an ablation.

---

## 2. Comparing text and finding duplicates

**Shingles (or n-grams)** **[used]** — overlapping windows of n words. "keep
rows where age is over" with n=3 gives "keep rows where", "rows where age",
"where age is", "age is over". Used because comparing individual words is too
crude — everything shares "the" and "return".

**Jaccard similarity** **[used]** — how much two sets overlap:
*shared ÷ total*. If two documents have 8 shingles in common out of 20 distinct
between them, that's 0.4. 1.0 means identical, 0.0 means nothing shared.

**MinHash** **[used]** — a trick for estimating Jaccard *fast*. Squash each
document into a fixed small fingerprint (say 128 numbers). Compare fingerprints
instead of documents. Necessary when you have hundreds of thousands of
documents and comparing every pair would take days. Cost: it's an estimate.

**LSH (locality-sensitive hashing)** — the next step up. Rather than comparing
every fingerprint to every other, it buckets similar ones together so you only
compare within buckets. What you reach for at millions of documents.

**Deduplication (dedup)** — removing duplicate and near-duplicate documents from
a corpus. Matters because training on the same text 50 times teaches the model
to parrot it. Code corpora are especially bad: vendored copies, forks, generated
files.

**Embedding** — a list of numbers representing a piece of text's *meaning*, so
that similar meanings land near each other. Catches paraphrases that word
overlap misses; slower, and needs a model to compute.

**Semantic vs lexical similarity** — lexical is "same words" (shingles,
Jaccard). Semantic is "same meaning" (embeddings). Lexical is cheap and literal;
semantic is expensive and fuzzy.

---

## 3. Training

**Pretraining** — the original, enormous training run where a model learns
language by predicting the next word over trillions of words. You will almost
certainly never do this; it costs millions.

**CPT — continued pretraining** **[used]** — more of that same next-word
training, but on *your* domain text. Teaches vocabulary and patterns. No
questions, no answers, just raw text. *Our plan: raw polars docs and code.*

**SFT — supervised fine-tuning** **[used]** — training on (instruction, correct
answer) pairs. Teaches the model to *respond*, in a particular format.

**DPO — direct preference optimization** **[used]** — training on (prompt,
better answer, worse answer) triples. Teaches preference between two outputs
that both look plausible. Simpler and cheaper than RLHF, and now more common.

**RLHF — reinforcement learning from human feedback** — the older approach:
humans rank outputs, you train a separate "reward model" to imitate their taste,
then train the main model against that. Powerful, fiddly, expensive.

**RLVR — RL with verifiable rewards** **[used]** — same idea, but the reward
comes from a *checker* rather than a human. Only possible where correctness is
automatically decidable — code that runs, maths that checks out. *Our verifier
is exactly this machinery.*

**Reward model** — a model trained to predict how much a human would like an
output. The thing RLVR lets you avoid.

**LoRA** **[used]** — instead of updating all the model's weights, train a
small set of extra ones alongside. Far cheaper, and the result is a small file
you can share. The default for anyone without a data centre.

**QLoRA** — LoRA on top of a compressed model, so it fits in less memory. What
makes training a 3B on a laptop plausible.

**Epoch** — one full pass over the training data. Three epochs means the model
saw everything three times.

**Learning rate** — how big a step to take on each update. Too high, training
explodes; too low, nothing happens. The most-tuned number in ML.

**Batch size** — how many examples the model looks at before updating. Bigger is
steadier and needs more memory.

**Loss** — a number saying how wrong the model currently is. Training tries to
push it down. **Important:** falling loss does *not* mean the model got better
at your actual task — that's what the eval is for.

**Precision / bf16 / 8-bit / 4-bit** **[used]** — how much detail each number
in the model is stored with. **bf16** is full precision (no compression);
8-bit is half the size; 4-bit a quarter.

**Quantization** **[used]** — storing the model's numbers roughly instead of
precisely. Smaller, faster, less memory — at the cost of some rounding. Like
saving a photo as a smaller JPEG: fine up to a point, then it looks bad.

**Q4_K_M / affine / AWQ / GPTQ / GGUF** **[used]** — different *recipes* for
that rounding, all producing "4-bit":
  - **affine** (MLX default) rounds everything the same way — simple, crude.
  - **Q4_K_M** (ollama) keeps important numbers precise, rounds the rest harder.
  - **AWQ** picks which numbers matter for real inputs and protects those.
  These are NOT interchangeable: on our task, affine 4-bit lost 10 points where
  Q4_K_M lost nothing (F27). Both are honestly labelled "4-bit".

**Quality regression** **[used]** — the model getting worse after a change that
was supposed to be neutral, e.g. compression. Always measure after quantizing.

**Loss masking / `--mask-prompt`** **[used]** — telling training "grade me only
on this part". You show the model the question and the answer, but score it only
on the answer, so it learns to write *answers* rather than to copy questions.
Like a student practising: you want them writing answers, not transcribing the
question. Get this wrong and the loss curve looks perfect while quality drops.

**Checkpoint** — a saved snapshot of the model mid-training.

**Distillation** — using a big expensive model to generate training data for a
small cheap one. *Exactly what we're doing: Claude writes the examples.*

**Catastrophic forgetting** — the model gets better at your new thing and worse
at everything it used to know. The main risk of aggressive fine-tuning, and a
reason to keep a general eval alongside your specific one.

---

## 4. Running models

**Inference** — using a trained model, as opposed to training it. Most
production cost is here.

**Token** — the chunk models actually read and write; roughly ¾ of a word.
Pricing and limits are in tokens.

**Context window** — how many tokens a model can consider at once.

**Prompt** **[used]** — the text you send. **Zero-shot** = just the instruction.
**Few-shot** **[used]** = the instruction plus worked examples. *Few-shot took
us from 20.8% to 35.4%.*

**System prompt** — standing instructions that apply to the whole conversation,
separate from the user's message.

**Temperature** **[used]** — randomness. 0 means always pick the most likely
word, so the same input gives the same output. *We use 0 so reruns are
comparable.*

**Greedy decoding** — the same idea: always take the most likely next token.

**Constrained decoding** — forcing output to match a grammar or schema, so it
*cannot* be malformed. Very useful for code and structured output.

**Latency** — how long one request takes. **p95 latency** **[used]** — the time
95% of requests come in under. Far more honest than the average, which one fast
path can flatter.

**Throughput** — how many requests you handle per second. Often trades against
latency.

**Batching / continuous batching** — processing several requests together for
efficiency. Continuous batching slots new requests in as others finish.

**KV cache** — the model's saved intermediate work for tokens it has already
read, so it doesn't redo it for every new token.

**Prefix caching** — reusing that saved work when many requests share the same
opening text (a long system prompt, say). Big win when prompts share a prefix.

**Quantization** **[used-later]** — storing weights with fewer bits (16 → 4) so
the model is smaller and faster. **AWQ**, **GPTQ**, **GGUF** are particular
schemes. Always check for quality regression afterwards — that's the whole risk.

**vLLM** — a popular high-performance serving engine; gives you continuous
batching, prefix caching and paged KV cache without writing them.

**Speculative decoding** — a small fast model guesses several tokens ahead and
the big model verifies them in one go. Faster with identical output.

---

## 5. Data

**Corpus** — a big pile of text used for training.

**Synthetic data** **[used]** — data generated by a model rather than collected
from the world. Cheap and plentiful; risks inheriting the generator's blind
spots.

**Verifier** **[used]** — code that decides whether an output is correct with no
human involved. *Ours runs the generated polars and compares the result.*

**Execution accuracy** **[used]** — scoring by *running* the code and comparing
results, rather than comparing the code as text. Correctly credits a right
answer written differently from yours.

**Quality filter** — a classifier that throws out low-quality documents before
training. Most scraped text is junk.

**PII — personally identifiable information** — names, emails, keys, addresses.
Must be stripped before training or publishing. Real repos are full of it.

**Provenance** — a record of where each piece of data came from and under what
licence. Boring until you want to publish, then essential.

**Data card / model card** — the documentation published alongside a dataset or
model: what's in it, how it was built, how it scores, what it's bad at. A good
one is a strong signal you know what you're doing.

**Golden set** — a small, carefully hand-checked set of examples treated as
ground truth. *Our 48 dev tasks.*

**Human-in-the-loop** — a person reviewing or correcting at some step, as
opposed to fully automatic.
