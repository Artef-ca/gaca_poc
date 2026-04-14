# GACA Sentiment Analysis Pipeline

Processes passenger feedback from TripAdvisor (airlines), Google Maps (airports),
and X/Twitter into dashboard-ready CSVs.

---

## How the Pipeline Works

Each data source goes through two LLM stages before being merged into a final dashboard file:

```
Raw data
   └─► [Step 0] Data Prep      — normalize, deduplicate, date-filter raw files
   └─► [Step 1] Sentiment      — Gemini extracts: sentiment, topic, pain points, moments of delight (free text)
   └─► [Step 2] Subtopic Map   — Gemini maps each free-text pain point / moment of delight → fixed taxonomy label
   └─► [Step 3] X Merge        — build X-only airline + airport outputs
   └─► [Step 4] Final Merge    — combine all sources → dashboard CSV
```

### What is the difference between Sentiment Extraction and Subtopic Mapping?

**Step 1 — Sentiment Extraction** reads a review and extracts:
- `sentiment` — Positive / Negative / Neutral
- `topic` — which GACA service area the feedback is about (e.g. *Boarding*, *Catering*, *Check-in*). Topics come from the taxonomy list.
- `pain_points` — **free-text** phrases describing what went wrong (e.g. *"long queues at gate"*, *"rude agent"*)
- `moments_of_delight` — **free-text** phrases describing what went well (e.g. *"fast boarding"*, *"friendly crew"*)

The pain points and moments of delight are open-ended — the LLM writes them in its own words.

**Step 2 — Subtopic Mapping** takes those free-text strings and maps each one to a **fixed label** from the taxonomy:

```
"long queues at gate"  →  subtopic: "Queue Management"
"rude agent"           →  subtopic: "Staff Attitude"
"fast boarding"        →  subtopic: "Boarding Efficiency"
```

This step is needed because the dashboard filters and aggregates by subtopic. Free text can't be counted or grouped — standardized labels can. The taxonomy CSV (`src/taxonomy_generation/topic_subtopic_refined.csv`) defines the full list of valid subtopics per topic.

---

## Project Structure

```
gaca_poc/
├── src/
│   ├── config.py                        # Central config: dates, model, paths, keywords
│   │
│   ├── data_prep/                       # One-time prep: normalize raw files for the pipeline
│   │   ├── prepare_tripadvisor.py       #   Combines + deduplicates TripAdvisor source CSVs
│   │   ├── prepare_google.py            #   Normalizes Google Maps scraper CSVs
│   │   └── prepare_x.py                #   Normalizes X/Twitter scraper xlsx/csv files
│   │
│   ├── data_pipelines/
│   │   ├── tripadvisor/                 # Airlines (TripAdvisor reviews)
│   │   │   ├── sentiment_extraction.py
│   │   │   └── subtopic_mapping.py
│   │   │
│   │   ├── google_maps/                 # Airports (Google Maps reviews)
│   │   │   ├── sentiment_extraction.py
│   │   │   └── subtopic_mapping.py
│   │   │
│   │   ├── x_data/                      # X/Twitter (airline + airport tweets)
│   │   │   ├── enrich.py                #   Classify + translate new tweets
│   │   │   ├── sentiment_extraction.py
│   │   │   ├── subtopic_mapping.py
│   │   │   ├── merge_airlines.py        #   Build X-only airline output
│   │   │   └── merge_airports.py        #   Build X-only airport output
│   │   │
│   │   ├── survey/                      # Survey pipeline (ready for future use)
│   │   │   ├── sentiment_extraction.py
│   │   │   └── subtopic_mapping.py
│   │   │
│   │   └── merge/                       # Final merge: all sources → dashboard CSV
│   │       ├── airlines.py              #   TripAdvisor + X → final airline CSV
│   │       └── airports.py              #   Google Maps + X → final airport CSV
│   │
│   ├── core/
│   │   ├── llm.py                       # Gemini helpers: make_model, retry, call-and-parse
│   │   └── language.py                  # Language detection + Gemini translation
│   │
│   ├── models/
│   │   ├── sentiment.py                 # Pydantic: ReviewsBatch, SubtopicsBatch
│   │   └── classification.py            # Pydantic: BatchClassification (tweet labelling)
│   │
│   ├── prompts/
│   │   └── loader.py                    # Loads prompts from config/*.yaml, injects placeholders
│   │
│   ├── utils/
│   │   └── helpers.py                   # combine_csvs, explode_column, fix_subtopic_format
│   │
│   └── taxonomy_generation/             # One-time scripts: generate + refine taxonomy
│       ├── generate_subtopics.py
│       └── refine_pp_mod.py
│
├── config/
│   ├── prompts.yaml                     # All LLM prompts (airline, airport, x, subtopic mapping)
│   └── survey_prompts.yaml              # Survey-specific prompts
│
├── raw_data/                            # ← Drop new scraper/source files here (gitignored)
│   ├── tripadvisor/
│   ├── google_maps/
│   └── x_data/
│
├── tripadvisor_data/                    # Processed TripAdvisor CSV — output of data_prep
├── Google_review_data/                  # Processed Google Maps CSV — output of data_prep
├── X_data/                              # Enriched X tweets + intermediate batches
├── airlines_sentiment/                  # Intermediate + final airline outputs
├── airport_sentiment/                   # Intermediate + final airport outputs
│
├── requirements.txt
├── .env                                 # API keys (not committed)
└── .gitignore
```

---

## Setup

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

> On Windows with Application Control policies, always use `python -m pip` instead of `pip` directly.

### 2. Set your API key

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
```

### 3. Set the batch date

Open `src/config.py` and update before each run:

```python
BATCH_DATE = '2026-04-14'   # today's date — used for intermediate folder naming
MIN_DATE   = '2025-11-01'   # earliest review date to include in output
```

---

## Running the Full Pipeline

All commands run from the **project root**.

### Step 0 — Prepare raw data

Run once when new source files arrive. Combines, deduplicates, and date-filters raw files.

```bash
python -m src.data_prep.prepare_tripadvisor   # reads raw_data/tripadvisor/*.csv
python -m src.data_prep.prepare_google         # reads raw_data/google_maps/*.csv
python -m src.data_prep.prepare_x              # reads raw_data/x_data/*.xlsx and *.csv
```

To pass specific files:
```bash
python -m src.data_prep.prepare_tripadvisor raw_data/tripadvisor/Fact_Reviews_new.csv raw_data/tripadvisor/Fact_Reviews_v2.csv
```

### Step 0b — Enrich new X tweets (incremental)

When you receive a new X/Twitter scraper export:

1. Open `src/data_pipelines/x_data/enrich.py` and add the new file to `NEW_FILES`:
   ```python
   NEW_FILES = [
       'X_data/Twitter Search Results Scraper_20260414.csv',
   ]
   ```
2. Run:
   ```bash
   python -m src.data_pipelines.x_data.enrich
   ```
   This classifies tweets (AIRLINE / AIRPORT), detects language, translates non-English,
   deduplicates against existing IDs, and appends to `X_data/final_combined_x_data.csv`.

### Step 1 — Sentiment extraction

Calls Gemini in batches. Already-processed batches are skipped automatically (checkpoint files).

```bash
python -m src.data_pipelines.tripadvisor.sentiment_extraction
python -m src.data_pipelines.google_maps.sentiment_extraction
python -m src.data_pipelines.x_data.sentiment_extraction
```

### Step 2 — Subtopic mapping

Maps free-text pain points and moments of delight to fixed taxonomy labels.

```bash
python -m src.data_pipelines.tripadvisor.subtopic_mapping
python -m src.data_pipelines.google_maps.subtopic_mapping
python -m src.data_pipelines.x_data.subtopic_mapping
```

### Step 3 — Build X outputs

```bash
python -m src.data_pipelines.x_data.merge_airlines
python -m src.data_pipelines.x_data.merge_airports
```

### Step 4 — Final merge (all sources → dashboard CSVs)

```bash
python -m src.data_pipelines.merge.airlines
python -m src.data_pipelines.merge.airports
```

**Output files:**
- `airlines_sentiment/final_airline_sentiment_{today}.csv`
- `airport_sentiment/final_airport_sentiment_{today}.csv`

---

## Adding New Data (Recurring)

No path changes needed — just drop files into the correct folder:

| New data | Drop here | Then run |
|---|---|---|
| TripAdvisor export | `raw_data/tripadvisor/` | `prepare_tripadvisor` → Step 1–4 |
| Google Maps scraper CSV | `raw_data/google_maps/` | `prepare_google` → Step 1–4 |
| X/Twitter scraper CSV | Edit `NEW_FILES` in `enrich.py` | Step 0b → Step 1–4 |

---

## Configuration Reference (`src/config.py`)

| Setting | Description |
|---|---|
| `BATCH_DATE` | Date string for intermediate folder names — **update each run** |
| `MODEL_NAME` | Gemini model (`gemini-2.5-flash`) |
| `MIN_DATE` | Earliest review/tweet date included in final output |
| `TAXONOMY_PATH` | Master topic/subtopic CSV used by all pipeline stages |
| `RAW_TRIPADVISOR_DIR` | Folder for TripAdvisor source files |
| `RAW_GOOGLE_DIR` | Folder for Google Maps scraper files |
| `RAW_X_DIR` | Folder for X/Twitter scraper files |
| `TRIPADVISOR_PATH` | Processed TripAdvisor CSV (output of Step 0, input to Step 1) |
| `GOOGLE_PATH` | Processed Google Maps CSV (output of Step 0, input to Step 1) |
| `X_COMBINED_PATH` | Enriched X tweet CSV (output of Step 0b, input to Step 1) |
| `LANG_MAP` | Language code → full name mapping |
| `AIRLINE_ENTITY_KEYWORDS` | Keyword rules for classifying airline tweets |
| `AIRPORT_ENTITY_KEYWORDS` | Keyword rules for classifying airport tweets |

---

## Environment Variables (`.env`)

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Google Gemini API key — required for all pipeline steps |

---

## Key Design Decisions

- **Batch checkpointing** — each batch is saved as its own CSV. If the pipeline crashes mid-run, re-running skips already-processed batches.
- **Incremental X enrichment** — `enrich.py` deduplicates by tweet ID before calling Gemini, so only truly new tweets are processed.
- **Mixed date formats** — `prepare_tripadvisor.py` handles multiple source files with different date formats (`format='mixed'`) and always keeps the most recent version of a review when deduplicating.
- **Prompt safety** — prompts use `_inject()` (not Python `.format()`) to safely handle YAML prompts that contain JSON examples with curly braces.
- **SDK** — uses `google-genai` (new SDK). All Gemini calls go through `src/core/llm.py:make_model()`.
