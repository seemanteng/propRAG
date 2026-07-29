# Proposition Graph Construction

A standalone notebook (`construct_graph.ipynb`) that builds a proposition graph — entities, passages,
and three edge types — from an already-computed OpenIE extraction of HotpotQA, scoped to the gold
evidence passages for bridge-type questions.

It reimplements the graph-construction step described in the **PropRAG** paper (*PropRAG: Guiding
Retrieval with Beam Search over Proposition Paths*, Wang & Han, 2025 — [arXiv:2504.18070](https://arxiv.org/abs/2504.18070),
Section 5.1 / Appendix A.1) from scratch. **This repo does not import or run any code from the
PropRAG codebase** (`PropRAG.py`, `graph_beam_search.py`, or anything under `src/proprag`) — only
`torch`, `transformers`, `networkx`, `tqdm`, and the Python standard library are used. It does,
however, consume *data* produced by that codebase's offline OpenIE step (see Inputs below).

## What it builds

Three edge types, matching the paper's definitions:

1. **Entity Clique (hyper-edge) edges** — every pair of entities that co-occurs within the same
   proposition is connected; weight accumulates across repeated co-occurrences. Built deterministically
   from the OpenIE output (no LLM involved — this is an exact fact already present in the data).
2. **Passage Containment edges** — each passage is connected to every entity that appears in one of
   its propositions. Also deterministic.
3. **Synonymy edges** — connects entity strings that refer to the same real-world entity (e.g. "Saint
   Peter" / "St. Peter"). This is the one genuinely ambiguous step, so it's resolved with a local LLM
   (Gemma E4B) rather than plain string matching. Since checking all ~13k entities pairwise is
   infeasible, entities are first grouped into candidate blocks (entities sharing a significant word),
   and the LLM resolves synonymy within each block.

Output is a `networkx.Graph` with `passage` and `entity` typed nodes, and edges carrying
`hyperedge_weight` / `containment_weight` / `synonymy_weight` attributes (an entity pair linked by more
than one edge type keeps all of the relevant weights on the same edge).

## Scope

Restricted to the union of gold `supporting_facts` passages across all **bridge-type** questions in
the HotpotQA query set — not the full corpus. This gives a graph over roughly:

- 1,618 passages
- 8,055 propositions
- 13,460 unique entities
- ~3,700 LLM synonymy-resolution calls (after blocking)

Change `QUESTION_TYPE` in the config cell (e.g. to `"comparison"`) or adjust the filtering logic to
scope differently.

## Inputs

Two JSON files (paths configurable via `OPENIE_PATH` / `QUERIES_PATH` in the config cell; the
defaults point at their locations under `PropRAG/`):

| File | Default path | Contents |
|---|---|---|
| `openie_results_ner_meta-llama_llama-3.3-70b-instruct.json` | `PropRAG/outputs/hotpotqa/` | OpenIE output: `{"docs": [{"idx", "passage", "extracted_entities", "propositions": [{"text", "entities"}]}, ...]}`. Note: `idx` is not a short id — it's literally `"chunk-" + passage` (the full passage text, prefixed), so passage/entity node keys built from it are long strings. |
| `hotpotqa.json` | `PropRAG/reproduce/dataset/` | HotpotQA queries: `{"_id", "question", "answer", "type", "supporting_facts": [[title, sent_idx], ...], "context": [...]}` |

Both are produced upstream by PropRAG's offline OpenIE/indexing step (or the original HotpotQA
release, for the second file) — this notebook only reads them.

## Requirements

```
torch
transformers
bitsandbytes       # for 4-bit model loading
accelerate         # required by transformers' device_map
networkx
tqdm
```

A local checkpoint for a Gemma E4B model (path or Hugging Face hub id), loaded in 4-bit via
`bitsandbytes`. A CUDA GPU is required for a full run in reasonable time; a 12GB GPU is enough with
default settings (see Performance notes below).

## Setup

1. Confirm `OPENIE_PATH` / `QUERIES_PATH` in the config cell point at the two input JSON files
   (defaults assume the `PropRAG/` layout above — update them if yours differs).
2. Set `GEMMA_MODEL_PATH` in the config cell to your Gemma E4B checkpoint.
3. Run the notebook top to bottom. Cell 1 (imports/config), Cell 3-4 (load + scope data), Cell 5
   (deterministic edges), Cell 6 (blocking) all run in seconds. The LLM synonymy cell is the slow one.

## Config reference

| Variable | Purpose |
|---|---|
| `OPENIE_PATH` / `QUERIES_PATH` | Paths to the two input JSON files |
| `GEMMA_MODEL_PATH` | Local path or HF hub id of the Gemma E4B checkpoint |
| `QUESTION_TYPE` | Which HotpotQA question type to scope the graph to (default `"bridge"`) |
| `SYNONYMY_BLOCK_SIZE_CAP` | Max entities per LLM synonymy call; larger token buckets are chunked to this size |
| `SYNONYMY_BATCH_SIZE` | How many blocks to run per `generate()` call (GPU batching). Lower this if you hit OOM |
| `LLM_DRY_RUN_LIMIT` | Cap on blocks processed, for a quick sanity check before a full run. Set to `None` for the full run |
| `OUTPUT_DIR` | Where graph + audit outputs are written (default `bridge_gold_graph/`) |

## Running the LLM synonymy step

- **Batched.** Blocks are processed `SYNONYMY_BATCH_SIZE` at a time via `GemmaLLM.generate_batch()`,
  not one at a time — decoding a small model at batch size 1 leaves most of the GPU idle.
- **Checkpointed and resumable.** Progress is written to `OUTPUT_DIR/synonymy_audit.jsonl` and
  `failed_blocks.json` every 25 batches, after any OOM recovery, and at the end. Re-running the cell
  picks up from that checkpoint (already-completed blocks are skipped) rather than starting over —
  safe to stop and resume across sessions.
- **OOM-safe.** A CUDA out-of-memory error is caught, the cache is cleared, and the batch is retried
  as two half-size batches (recursively down to size 1) instead of crashing the run. If OOMs happen
  often, lower `SYNONYMY_BATCH_SIZE`.
- **Timed.** Each run prints a measured avg seconds/batch and an ETA for the blocks still remaining,
  based on that session's own measurements — start with a small `LLM_DRY_RUN_LIMIT` to get a real
  number for your hardware before committing to a full run.

## Outputs

Written to `OUTPUT_DIR` (default `bridge_gold_graph/`):

| File | Contents |
|---|---|
| `graph.gpickle` | The full `networkx.Graph`, pickled (primary artifact — preserves all attributes) |
| `graph.graphml` | Same graph, GraphML format with attributes stringified (for Gephi etc.) |
| `synonymy_audit.jsonl` | One line per processed block: input entities, raw LLM output, parsed clusters — for reviewing synonymy decisions and for resuming a run |
| `failed_blocks.json` | Blocks whose LLM response couldn't be parsed as valid JSON, or that OOM'd even at batch size 1 |

## Known limitations

- **Blocking is lossy.** Two entities that don't share any significant word never get compared, so a
  synonym pair with no lexical overlap (e.g. very different aliases) won't be found. Within an
  over-large token bucket, entities in different chunks also aren't cross-compared.
- **Synonymy disambiguation context is one example sentence per entity** (the first proposition it
  appeared in) — an entity with genuinely ambiguous mentions across the corpus only gets one data point
  to disambiguate with.
- **Greedy decoding (temperature 0)** is used for determinism, but synonymy quality still depends
  entirely on the underlying model's judgment — spot-check `synonymy_audit.jsonl` before trusting the
  resulting edges for anything downstream.

## Citation

If referencing the graph structure this notebook implements:

```
@article{wang2025proprag,
  title   = {PropRAG: Guiding Retrieval with Beam Search over Proposition Paths},
  author  = {Wang, Jingjin and Han, Jiawei},
  journal = {arXiv preprint arXiv:2504.18070},
  year    = {2025}
}
```
