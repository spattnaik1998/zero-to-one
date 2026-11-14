# Testing *Zero to One*

A claim-by-claim empirical test of Peter Thiel's *Zero to One* (published 16 September 2014). Each hypothesis takes one specific argument from the book, states what would count for and against it, and checks it against public data from before and after publication.

The full write-up, with figures and numbers, is in [`output/Zero_to_One_Findings.pdf`](output/Zero_to_One_Findings.pdf).

## Findings

| # | Claim (chapter) | Verdict | In one line |
|---|---|---|---|
| H1 | Power law: a few winners produce most of the value (7) | Partly holds | Wealth creation is extremely concentrated and has become more so since the book. The rule that one holding outweighs all the others fits the best portfolios, not typical ones. |
| H2 | Competition destroys profits; monopolies capture value (3–4) | Partly holds | Firms that dominate their market earn higher margins, and the premium has grown. Industry-wide concentration brings no premium, and dominating a small market pays less. What pays is dominance at scale. |
| H3 | Durability and last-mover advantage (5) | Holds, with a twist | Dominant leaders keep their markets three times in four. The twist: in 2014 the market underpriced the tech giants' durability, since 11 years of cash flow repaid their whole 2014 price. |
| H4 | The companies Thiel named (3, 5, 6, 12, 13) | Mostly holds | Every firm he held up as a monopoly or definite planner beat the index, and his warnings about airlines and Groupon came true. His New York Times vs. Twitter point reversed. |
| H5 | Indefinite optimism produces companies without plans (2, 6, 13) | Holds | Money-losing IPOs reached their highest sustained share on record, followed by a SPAC wave. Such firms rarely became profitable. In cleantech, the technology won while its investors trailed the market. |
| H6 | Stagnation: progress in bits, not atoms (1, conclusion) | Mostly holds | The split between bits and atoms is stark and has continued. Productivity growth has not returned to its 1948–73 pace. The pickup since 2022 is too early to call a takeoff. |
| H7 | Sales matters as much as product (2, 11) | Holds, with a caveat | Tech firms spend more on selling than on building, and sales spending is followed by faster growth than R&D is. Investors still pay more for R&D. |
| H8 | Computers complement humans (12) | Mostly holds | Since 2014 productivity, pay and employment all rose, and computer-heavy jobs grew fastest. Routine clerical work was replaced. So far AI exposure shows no job loss. |

## Notebooks and data

| Notebook (in `notebooks/`) | Claim | Data |
|---|---|---|
| `01_H1_power_law` | Power law | SEC XBRL public float and total assets; Y Combinator directory |
| `02_H2_competition_profits` | Competition destroys profits | Census Economic Census and ASM; SEC XBRL financials and SIC codes |
| `03_H3_durability` | Durability and last-mover advantage | SEC XBRL financials, cash flows and public float |
| `04_H4_named_companies` | Thiel's named companies since publication | Yahoo Finance; `reference/delisted_prices.csv`; deal terms from SEC filings |
| `05_H5_indefinite_optimism` | Indefinite optimism | Ritter IPO statistics (`reference/ritter_*.csv`); SEC XBRL; Our World in Data; Yahoo Finance |
| `06_H6_stagnation` | Stagnation | FRED (BLS productivity and CPI components, BEA investment, Penn World Table, World Bank) |
| `07_H7_sales_vs_product` | Sales vs. product | SEC XBRL sales-and-marketing, R&D, revenue, public float |
| `08_H8_computers_complement` | Computers complement humans | FRED; BLS OEWS occupational employment and wages; O\*NET work activities; AI Occupational Exposure (AIOE) |

## Setup

Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use `source .venv/bin/activate` elsewhere
pip install -r requirements.txt
```

Create a `.env` file in the repo folder or any folder above it:

```
SEC_USER_AGENT=Your Name you@example.com   # SEC requires a contact; also sent to BLS
CENSUS_API_KEY=...                         # https://api.census.gov/data/key_signup.html
FRED_API_KEY=...                           # https://fred.stlouisfed.org/docs/api/api_key.html
```

## Running

Run the notebooks in order (Kernel → Restart & Run All), or headless:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_H1_power_law.ipynb
```

Then rebuild the report:

```bash
python findings_report.py
```

Downloads are cached in `data/raw/` (not committed). The first run of each notebook fetches its data; later runs read the cache. Delete a file there to refresh it. If a site refuses a scripted download, the notebook prints the URL and the path to save the file to by hand.

To exercise the code without network access or keys, set `THIEL_SYNTHETIC=1`. The notebooks then run on generated data; the results are meaningless and are for testing only.

## Ask Zero to One (in development)

An agentic RAG assistant that answers questions about the book's arguments with cited evidence from three sources: the book, the analyst's reading notes, and the findings above. Claude handles reasoning and OpenAI handles embeddings. The design documents are not published here; [`docs/README.md`](docs/README.md) describes them and summarises the architecture.

```bash
pip install -r requirements-rag.txt
cp .env.example .env      # or add the new keys to your existing .env (see the note inside)
python -m rag doctor      # checks keys, models, sources and analysis outputs
pytest -q
```

The book and the notes go in `sources/book/` and `sources/notes/`. Neither is committed; see [`sources/README.md`](sources/README.md).

## How it fits together

- **`thiel_common.py`** holds all shared code: data fetchers with on-disk caching, SEC data cleaning (screening public float against total assets, fixing unit-scale errors, linking renamed or acquired companies), statistics helpers (power-law tail fits, fixed-effects OLS with clustered standard errors, weighted least squares), and the plot style.
- **Each notebook** quotes its claim, runs the tests, saves figures to `output/figures/` and tables to `output/tables/`, and ends with a check cell that writes `output/hN_summary.json`.
- **`findings_report.py`** builds the PDF from those summary files and figures only, so it can be re-run at any time. Hypotheses whose notebook has not run yet are listed as pending.
- **`reference/`** holds inputs that cannot be downloaded by script (Ritter IPO tables, prices for delisted stocks).
