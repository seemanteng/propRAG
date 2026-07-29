# Two-Stage Graph Retrieval

A standalone notebook (`retrieval_mechanism.ipynb`) that answers HotpotQA-style queries against the
proposition graph built by `construct_graph.ipynb`, using a two-stage retrieval pipeline: personalized
PageRank for coarse recall, then beam search over proposition paths for precision. **No LLM calls
anywhere in this pipeline** — it's embeddings (BGE-M3, dense-only) + PageRank + beam search only.

It reimplements the retrieval step described in the **PropRAG** paper (*PropRAG: Guiding Retrieval
with Beam Search over Proposition Paths*, Wang & Han, 2025 —
[arXiv:2504.18070](https://arxiv.org/abs/2504.18070)) on top of the graph from `construct_graph.ipynb`.
It does not import or run any code from the PropRAG codebase — only `FlagEmbedding`, `networkx`,
`numpy`, `tqdm`, and the standard library are used.

## How it works

### Setup (once)

Every proposition across all in-scope passages is embedded once with a local **BGE-M3** checkpoint
(dense vectors only — sparse/ColBERT-style scoring is not used) and cached to disk, so re-running the
notebook doesn't re-embed from scratch. Each graph edge also gets a single computed `weight` attribute
= the sum of whichever of `hyperedge_weight` / `synonymy_weight` / `containment_weight` are present, so
`nx.pagerank` (which only accepts one weight attribute name) can run over the mixed-edge-type graph.

### Stage 1 — coarse retrieval (whole graph)

1. Embed the query and cosine-rank it against every proposition embedding → top-N propositions.
2. Seed a personalization vector from the entities in those propositions (`Sinitial`).
3. Run personalized PageRank over the **whole graph** from that seed.
4. Take the top-K passages by PPR score, and induce a local subgraph (`G_sub`) around them: passages →
   their entities (containment) → those entities' neighbors (hyperedge/synonymy).

### Stage 2 — beam search (`G_sub` only, no PageRank)

Starting from the top query-similar propositions whose entities land in `G_sub`, extend each beam path
hop by hop — candidates come from entities shared with other propositions (graph-connected) plus a
handful of pure top-similarity "jump points" that ignore graph connectivity, so a broken or missing
graph link can't stall the search. Paths are scored by cosine similarity between the query and the
mean embedding of the path's propositions; only the top `beam_width` paths survive each hop.

### Final scoring

Entities on surviving beam paths accumulate `entity_scores` (with an extra boost across
synonymy-linked consecutive propositions), combined with `Sinitial` into `Sfinal`. A second,
`G_sub`-local PageRank run from `Sfinal` produces the final passage ranking.

## Scope

Runs over the same bridge-type query scope used to build the graph — 811 queries, confirmed by count
against `hotpotqa.json` before the batch run. Change `QUESTION_TYPE` / `EXPECTED_NUM_QUERIES` in the
config cell if you're pointing this at a different scope.

## Inputs

| File | Contents |
|---|---|
| `bridge_gold_graph/graph.gpickle` | The proposition graph from `construct_graph.ipynb` — passage/entity nodes, hyperedge/synonymy/containment-weighted edges |
| `openie_results_ner_meta-llama_llama-3.3-70b-instruct.json` | OpenIE output: `{"docs": [{"idx", "passage", "propositions": [{"text", "entities"}]}, ...]}` |
| `hotpotqa.json` | HotpotQA queries: `{"_id", "question", "answer", "type", "supporting_facts": [[title, sent_idx], ...]}` |

## Requirements

```
FlagEmbedding
networkx
numpy
tqdm
```

A local **BGE-M3** checkpoint (path or Hugging Face hub id). No GPU is strictly required for
retrieval-time embedding (a single query at a time), but one speeds up the one-time proposition
embedding pass considerably.

## Setup

1. Run `construct_graph.ipynb` first (or otherwise produce `bridge_gold_graph/graph.gpickle`).
2. Set `BGE_M3_MODEL_PATH` in the config cell to your BGE-M3 checkpoint.
3. Run the notebook top to bottom. Sections 1-4 (load, embed model, proposition index, edge weights)
   run in seconds except the first-time proposition embedding pass. Section 8a runs a single-query
   smoke test — check it looks sane before Section 9's full 811-query batch run.

## Config reference

| Variable | Purpose |
|---|---|
| `GRAPH_PATH` / `OPENIE_PATH` / `QUERIES_PATH` | Paths to the graph and the two input JSON files |
| `BGE_M3_MODEL_PATH` | Local path or HF hub id of the BGE-M3 checkpoint |
| `BGE_M3_BATCH_SIZE` | Batch size for proposition/query embedding calls |
| `BGE_M3_QUERY_INSTRUCTION` | Optional instruction prefix for query embeddings (default empty — BGE-M3 doesn't require one) |
| `QUESTION_TYPE` / `EXPECTED_NUM_QUERIES` | Query scoping rule + expected count, asserted before the batch run |
| `N_PROP_STAGE1` | Top-N propositions by similarity used to seed stage-1 PPR |
| `TOP_K_PASSAGES` | Top-K passages by stage-1 PPR score, kept for the beam-search subgraph |
| `PPR_ALPHA_STAGE1` / `PPR_ALPHA_STAGE2` | PageRank damping factor for the coarse (whole-graph) and final (subgraph) PPR runs |
| `SUBGRAPH_EXPANSION_HOPS` | Radius of the induced subgraph around the top-K passages |
| `BEAM_WIDTH` | Number of paths kept after each beam-search hop |
| `MAX_BEAM_HOPS` | Number of beam-search extension hops |
| `NUM_JUMP_POINTS` | Top-similarity propositions injected at every hop regardless of graph connectivity |
| `SYNONYMY_BRIDGE_BOOST` | Extra additive entity-score boost across synonymy-linked consecutive path propositions |
| `TOP_K_RESULTS` | Number of passages returned per query |
| `RECALL_AT_K_VALUES` | K values checked in the evaluation cell |

## Running the batch retrieval step

- **Checkpointed and resumable.** Each query's result is appended to `retrieval_results.jsonl`
  immediately after it's computed. Re-running the batch cell skips any `query_id` already present in
  that file, so stopping and resuming across sessions is safe.
- **Failures isolated.** A per-query exception is caught, logged (with the query id and error) to
  `retrieval_failures.jsonl`, and the batch continues rather than crashing on one bad query.
- **Timed.** Prints a running average seconds/query and a rough ETA after the first 20 queries of a
  given run, so a bad ETA is visible early rather than after the full 811-query run.

## Outputs

| File | Contents |
|---|---|
| `bridge_gold_graph/prop_embeddings.npy` | Cached dense proposition embeddings (one-time cost) |
| `bridge_gold_graph/prop_meta.json` | Parallel metadata: `{prop_id, pid, entities, text}` per embedding row |
| `retrieval_results.jsonl` | One line per query: `{query_id, question, retrieved: [{pid, score, text}, ...]}` |
| `retrieval_failures.jsonl` | One line per failed query: `{query_id, question, error}` |

## Known limitations

- **Beam search only sees `G_sub`.** If stage 1's top-K passages miss the actual answer passage
  entirely, stage 2 can't recover it — the two stages are not independent.
- **Jump points are a blunt fallback** for broken graph connectivity; they're drawn from
  whole-corpus similarity, not `G_sub`, so they can pull in propositions whose entities aren't even in
  the current subgraph (path embedding score still applies, but the "path" may not be graph-coherent).
- **Recall@K evaluation is title-based**, matching retrieved passages to gold `supporting_facts` by
  title — it's a coarse sanity check, not a substitute for a proper answer-level eval.

## Citation

If referencing the retrieval method this notebook implements:

```
@article{wang2025proprag,
  title   = {PropRAG: Guiding Retrieval with Beam Search over Proposition Paths},
  author  = {Wang, Jingjin and Han, Jiawei},
  journal = {arXiv preprint arXiv:2504.18070},
  year    = {2025}
}
```
