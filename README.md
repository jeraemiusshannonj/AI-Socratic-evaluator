# AI Socratic Evaluator

A debugging tutor that generates Python code containing one deliberate logic
bug, then helps a student find it by asking rather than telling. It runs the
student's fix against executable test cases before saying anything about it,
and grounds its explanations in passages retrieved from a CS corpus, citing
the ones it used.

Built on **MongoDB Atlas Vector Search** and **Google Vertex AI**
(`gemini-3.8-flash`, `gemini-embedding-001`).

---

## The part that is actually interesting

Two retrieval pipelines, kept separate because they answer different kinds
of question.

**Semantic, over unstructured prose.** "Why does `high = mid` loop forever?"
has an answer that must be read and understood. A `$vectorSearch` over 504
embedded chunks of CS textbook material returns the passages that explain the
mechanism, optionally scoped to a concept.

**Structured, over records.** "Which concepts does this student fail most
often?" is a computation, not a search. No embedding can answer it. Ordinary
Atlas aggregations over the misconception taxonomy, 115 interaction records
and 79 test cases serve those.

They are separate, individually callable functions in
[`socratic/retrieval.py`](socratic/retrieval.py), split into `SEMANTIC_TOOLS`
and `STRUCTURED_TOOLS`, so a routing decision is visible in a trace rather
than hidden inside one query that gets called hybrid.

**Correctness comes from running the code, not from reading it.** Whether a
fix works is a question with an answer, and reading code to guess at it is
what language models are least dependable at. Every evaluation runs the
student's code against that concept's test cases first; the result is the
verdict, and the model's own reading is recorded beside it for comparison.

**Citations are checked.** The critic is given passages with ids and asked
which it used. Every cited id is verified against what was actually supplied,
so an invented source is recorded as invalid rather than believed.

---

## Measured

From `python -m scripts.run_eval --all`. The full report is
[`eval/results/run.json`](eval/results/run.json).

### Retrieval

28 questions covering all 22 concepts, each naming the specific documents
that could honestly answer it (2.4 on average). Run twice:

| | hit@1 | hit@5 |
|---|---|---|
| with concept filter | **93%** (26/28) | 100% (28/28) |
| without concept filter | **89%** (25/28) | 100% (28/28) |

Median query latency **190 ms**, embedding included.

The unfiltered row is the one worth reading. Dropping the metadata filter
costs four points, which says the embeddings are carrying the retrieval and
the concept tags are a refinement rather than a crutch. An earlier version of
this eval matched on repository prefix instead of document, which with the
filter applied was nearly free; it reported 90% and was measuring almost
nothing.

### Critic

Over 44 reference implementations — one correct and one carrying the
documented misconception, for each of the 22 concepts
([`eval/references.py`](eval/references.py)).

| | |
|---|---|
| verdict correct | 44/44 |
| the model's reading agreed with the test run | 44/44 |
| cited at least one passage it was given | 41/44 |
| citations made | 52 |
| citations referring to a passage never supplied | **0** |

Verdict accuracy here is mostly a statement about the test cases, since the
verdict comes from them. The agreement figure is the interesting one, and it
needs a caveat: these 44 implementations are deliberately unambiguous. Real
student code is messier, and 100% agreement on clean cases does not predict
100% on ambiguous ones. The honest claim is that it agreed on all 44
unambiguous cases.

### Cost and latency

| | |
|---|---|
| tokens per evaluation | ~1,740 |
| throughput | 76.6 tokens/sec |
| p50 latency | 5.7 s |
| p95 latency | **50.7 s** |

That p95 is bad and is a known problem — see [Known
issues](#known-issues).

### Corpus

| | |
|---|---|
| documents | 54 |
| chunks | 504 |
| median chunk | 1,305 characters |
| embedding | `gemini-embedding-001`, 768 dimensions, cosine |
| ingest throughput | 17.2 chunks/sec |
| licences | MIT, PSF-2.0 |

---

## Corpus and licensing

The repository is public, so every source is permissively licensed and the
check was done before ingesting rather than after.

| Source | Licence | Covers |
|---|---|---|
| [OpenDSA](https://opendsa.org) | MIT | searching, sorting, hashing, graphs, recursion, ADTs |
| [CPython documentation](https://docs.python.org/3/) | PSF-2.0 | Python language semantics |
| [SQLAlchemy documentation](https://www.sqlalchemy.org/) | MIT | SQL joins and grouping |

Two decisions worth recording.

**PY4E is excluded.** OpenDSA vendors the *Python for Everybody* chapters,
but those are Charles Severance's under CC BY-NC-SA — OpenDSA cannot
relicense someone else's work by including it. The non-commercial and
share-alike terms are both unwanted on a public repository, so the Python
concepts are served from CPython's own documentation instead. That turned out
to be the better source anyway: `tutorial/controlflow.rst` documents the
mutable-default-argument trap directly, which is one of the misconceptions
this app teaches.

**OpenDSA's SQL chapter is unusable.** It is entirely interactive JavaScript
slideshows; `SQLJoin.rst` is three directives and a heading and converts to 18
characters of text. SQLAlchemy's narrative documentation covers both SQL
concepts with the rendered SQL shown alongside the Python.

**The converted corpus is not committed.** `corpus/sources.py` plus
`scripts/build_corpus.py` reproduce it exactly, so the repository carries the
method rather than a copy of three documentation projects.

---

## How retrieval is built

**Structure-aware chunking** ([`socratic/chunking.py`](socratic/chunking.py)).
Fixed-width splitting cuts code blocks in half and strands explanations from
the heading that says what they explain; retrieval quality then looks like a
model problem when it is a chunking problem. This keeps fenced code intact,
splits on heading boundaries first, prefixes each chunk with its heading trail
so the embedding carries the context a reader would have had from scrolling
up, and hard-splits anything approaching the embedding input ceiling so no
chunk is ever silently embedded as a partial.

**Chunk-level concept filtering** ([`corpus/signals.py`](corpus/signals.py)).
Document-level tags are right for "this page is about hashing" and wrong for a
vector-index filter. CPython's programming FAQ is a legitimate source for
three Python concepts and is also 56 chunks, most about threading and string
formatting. Tagging all 56 lets a search scoped to Python Lists return a
passage with nothing to cite — which reads as a bad embedding model but is bad
metadata. Each chunk now earns its tags through evidence: one distinctive
phrase, or several weaker terms together. This cut Python Lists from 139
candidate chunks to 45 and SQL Left Join from 102 to 33. Chunks that earn
nothing keep an empty tag list and stay reachable by unfiltered search.

**Asymmetric embedding.** Documents are embedded with
`RETRIEVAL_DOCUMENT` and queries with `RETRIEVAL_QUERY`. A question and the
passage that answers it are not the same kind of text, and using one task
type for both measurably degrades recall.

**Idempotent ingest.** Deterministic `_id`s, upserts, and a stored content
hash alongside the embedding model and dimension, so a re-run embeds only what
changed. Metadata is written separately from vectors: widening a signal set
changes `concept_tags` without changing a character of text, and those tags
have to reach the database without paying to re-embed.

---

## Test cases

79 cases across all 22 concepts, in
[`data/test_cases.json`](data/test_cases.json). Each concept has at least one
case written to fail on that concept's specific misconception and pass on a
correct implementation, plus ordinary correctness cases so a fix that breaks
something else is not read as success.

Three of the misconceptions return perfectly correct output and differ only in
work done — a bubble sort missing its early exit, a selection sort swapping in
the inner loop. Those are caught by a list subclass that counts reads and
writes. The binary-search off-by-one does not return anything wrong either; it
hangs, so that case runs in its own process and is judged on whether it
terminates at all.

`python -m scripts.validate_test_cases` runs every case against both reference
implementations: the correct one must pass everything, and the buggy one must
fail the case written for it. That check found five problems in the first
draft. Four cases compared a tuple against a JSON list and so failed *correct*
implementations. More usefully, every heap-sort case passed the buggy
implementation, because 1-indexed child arithmetic still sorts about 80% of
random inputs — the hand-picked examples all landed in the 80%. `[1, 2, 3]` is
the smallest input that distinguishes them.

---

## Running it

Needs a Google Cloud project with the Vertex AI API enabled and a MongoDB
Atlas cluster. The free M0 tier is sufficient: vector search works on it, and
this uses one of its three allowed search indexes.

```bash
cp .env.example .env          # set GCP_PROJECT and MONGO_URI
gcloud auth application-default login

python -m scripts.check_models      # verify the model IDs resolve
python -m scripts.bootstrap_db      # collections and the vector index
python -m scripts.migrate_sqlite    # taxonomy and interaction history
python -m scripts.build_corpus      # fetch and convert the upstream sources
python -m scripts.ingest_corpus     # chunk, embed, upsert
python -m scripts.seed_test_cases   # load the test cases
```

Then, in two terminals:

```bash
uvicorn main:app --reload --port 8000
streamlit run app.py
```

`GET /api/health` checks Mongo, the chunk count and whether the vector index
is queryable, so a green response means the whole chain is live.

Inspect retrieval without the UI:

```bash
python -m scripts.try_search "why does high = mid loop forever"
python -m scripts.try_search --structured "Binary Search (Off-by-One)"
python -m scripts.run_eval --retrieval
```

---

## Known issues

**`/api/run` executes arbitrary Python and is not a security boundary.**
[`socratic/execute.py`](socratic/execute.py) runs student code in isolated
mode with a fresh working directory, a minimal environment (so the Mongo
connection string is not inherited), output truncation, a wall-clock timeout
and POSIX resource limits. That makes accidents survivable. It does nothing
against someone trying: it would stop neither `open('/etc/passwd').read()` nor
an outbound network call. **Do not expose this service publicly without a real
boundary** — a container with no network and a read-only root, gVisor, or a
hosted execution service.

**p95 latency is 50.7 seconds** against a p50 of 5.7. Two causes: Gemini 3.x
reasons before answering with no budget currently set, and one call in the
measured run hit a 429 and backed off. The embedding call and the taxonomy
lookup also run serially when they could overlap. None of this is addressed
yet.

**72 of the 115 interaction records cannot be attributed to a specific
misconception.** They are logged against a bare `"Binary Search"`, which is not
a concept in the taxonomy — only `Binary Search (Off-by-One)` and
`Binary Search (Loop Condition)` are. The migration keeps the original label
and adds a `concept_family` so they can be rolled up, but no per-variant claim
can be made about the largest block of real data here.

**The original schema had no student identifier**, so all 112 migrated rows sit
under a single `legacy` id and support no per-student claim. Rows written since
carry one.

**Chunk ids are derived from filenames, not sections.** A passage about
sequential search cites as `opendsa:Searching_BinarySearch#0`, because OpenDSA
puts both algorithms in one file. The citation is correct and reads as though
it is not.

**`MIN_PASSAGE_SCORE` is untuned.** It is set to 0.45 and observed scores
cluster around 0.82–0.84, so it never fires. For "the corpus has nothing
relevant" to ever trigger, that threshold needs to come from measurement.

---

## Layout

```
socratic/          the application
  retrieval.py     both pipelines, as separately callable tools
  tutor.py         exercise generation and Socratic evaluation
  verify.py        runs test cases against student code
  chunking.py      structure-aware splitter
  vertex.py        all model access, retries, token accounting
  db.py            Atlas collections and index bootstrap
  execute.py       code execution, and what it is not
  history.py       interaction logging
corpus/            build-time only; nothing here runs at request time
  sources.py       which documents, which concepts, which licence
  signals.py       per-chunk concept evidence
  rst.py           reStructuredText to Markdown, without Sphinx
eval/              ground truth
  references.py    correct and buggy implementations of all 22 concepts
  retrieval_questions.json
scripts/           setup, ingest, and measurement
data/              test_cases.json
```
