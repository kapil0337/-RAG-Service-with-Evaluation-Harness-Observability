# RAG Evaluation Results

> **Note:** This report contains demo data. A live run (`make eval` /
> `python -m evals.run_eval`) requires a running Postgres + pgvector
> instance and a `GROQ_API_KEY` with available daily quota - the
> configured key's free-tier quota was exhausted during development.
> The numbers below are representative values for a healthy run against
> this corpus and should be replaced by re-running `make eval` once a
> live environment is available.

Evaluates **13** questions (from `evals/dataset.json`, including 2
intentionally unanswerable questions) against the corpus in `sample_docs/`.
LLM: `llama-3.1-8b-instant` | Embeddings: `BAAI/bge-small-en-v1.5`

## Aggregate Scores

| Metric | Score |
|---|---|
| faithfulness | 0.943 |
| answer_relevancy | 0.898 |
| context_precision | 0.794 |
| abstention_accuracy | 1.000 |

## Per-Question Results

| # | Question | Answer | faithfulness | answer_relevancy | context_precision | abstained |
|---|---|---|---|---|---|---|
| 1 | How many days of annual leave do Atlas Dynamics employees get? | Atlas Dynamics employees receive 21 days of annual leave per calendar year, in addition to 12 public holidays. | 0.950 | 0.920 | 0.880 | no (n/a) |
| 2 | What is the notice period after probation is completed? | The notice period after completing probation is 60 days. | 0.970 | 0.940 | 0.850 | no (n/a) |
| 3 | Which days are employees expected to be in the office? | Employees are expected to be in the office on Tuesday, Wednesday, and Thursday. | 0.960 | 0.930 | 0.900 | no (n/a) |
| 4 | How much is the annual learning budget per employee? | Each employee has an annual learning budget of INR 40,000 per year. | 0.940 | 0.910 | 0.860 | no (n/a) |
| 5 | What is the maximum payload of the Aurora X1? | The Aurora X1 has a maximum payload capacity of 50 kg. | 0.960 | 0.950 | 0.910 | no (n/a) |
| 6 | How long does the Aurora X1 battery last, and how long does it take to fully charge? | The Aurora X1's battery provides 8 hours of runtime per full charge and takes 90 minutes to fully charge. | 0.930 | 0.920 | 0.840 | no (n/a) |
| 7 | How much does one Aurora X1 unit cost? | One Aurora X1 unit costs INR 12,00,000. | 0.970 | 0.940 | 0.890 | no (n/a) |
| 8 | How many Aurora X1 units can a single Atlas Fleet Controller coordinate? | A single Atlas Fleet Controller can coordinate up to 20 Aurora X1 units. | 0.950 | 0.930 | 0.870 | no (n/a) |
| 9 | In what year was Atlas Dynamics founded and where is it headquartered? | Atlas Dynamics was founded in 2021 and is headquartered in Bengaluru, India. | 0.960 | 0.940 | 0.900 | no (n/a) |
| 10 | What is the return window for products? | Products can be returned within 30 days of delivery, provided they are in their original condition. Custom-configured units are non-returnable. | 0.940 | 0.910 | 0.830 | no (n/a) |
| 11 | What is the warranty period on Atlas Dynamics hardware? | Atlas Dynamics hardware comes with a standard 2-year warranty, which can be extended up to 5 years. | 0.950 | 0.920 | 0.790 | no (n/a) |
| 12 | Can the Aurora X1 operate underwater? | The provided documents do not mention whether the Aurora X1 can operate underwater, so I cannot answer this based on the available information. | 0.880 | 0.750 | 0.420 | yes |
| 13 | What is Atlas Dynamics' policy on stock options for employees? | The documents do not provide information about a stock options policy for employees. | 0.900 | 0.710 | 0.380 | yes |
