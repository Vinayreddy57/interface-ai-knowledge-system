# Evidence-first snapshot knowledge system

`solution.py` exposes `build()`, and the required `ingest`, `answer` and `conflicts` methods. Each snapshot is stored separately, so `as_of` chooses the most recent snapshot no later than the requested date; unqualified questions use the latest snapshot.

The retrieval index operates on page titles, section headings, table rows and short text windows, with exact-match and lexical ranking. Official product, rate, and branch pages are preferred to old blog posts and calculators. We preserve table headers and row qualifiers and decline unsupported questions. Citations are returned as `Source` objects. Conflict extraction compares numerical facts for a particular URL, product and tier across snapshots; it also handles short narrative rate lines. It avoids treating different balance tiers as conflicting. It currently prioritizes changed facts across snapshots, and does not exhaustively detect cross-page contradictions.

No external API, credentials, model downloads, paid services or network access are required. Python 3.11 with `scikit-learn` and `numpy` is sufficient. Runtime costs are dominated by ingestion/index building; individual answers search a prebuilt sparse index.

Testing: `python harness/run_eval.py --submission my_solution --eval eval/public_eval.jsonl`

Limitations: lexical search cannot always disambiguate subtle implied financial products; multiple true tiers require an explicit selection. Where the source cannot support a reliable answer the system abstains. The participant website snapshots are intentionally excluded from the public submission; the evaluation harness supplies them separately.
