# Data Card

## Overview

This document records the datasets bundled in this repository, their structure, and — importantly — their provenance and licensing, which are **separate from the project's code license**. The repository's `LICENSE` (MIT) covers the source code only; it does **not** grant any rights over the third-party data files under `src/data/`. The corpora here are small, label-only sentiment sets with no direct personal identifiers (no usernames, user IDs, timestamps, or geolocation), so the privacy surface is low, but redistribution terms depend on each set's original license. Source and license fields below are marked _to confirm_ where the upstream origin was not recorded at ingestion time; fill them from your download record before treating redistribution as cleared.

## Table of Contents

- [Overview](#overview)
- [Datasets](#datasets)
- [Source & license](#source-license)
- [Privacy](#privacy)

## Datasets

| File | Rows | Columns | Role |
|------|-----:|---------|------|
| `src/data/raw/1k_data_emoji_tweets_senti_posneg.csv` | 1,000 | index, `sentiment`, `post` | Raw labeled tweet corpus (training/eval) |
| `src/data/raw/15_emoticon_data.csv` | 16 | index, `Emoji`, `Unicode codepoint`, `Unicode name` | Emoji reference lexicon |
| `src/data/processed/tweets_clean.csv` | 1,000 | `label`, `text` | Cleaned tweet corpus (derived) |
| `src/data/processed/features_final.csv` | — | engineered features | Model input matrix (derived) |
| `src/data/processed/emoji_reference_clean.csv` | — | cleaned lexicon | Derived from the emoticon reference |

Derived files are reproducible from the raw inputs via `make train` (data cleaning → feature engineering).

## Source & license

> Fill these in from where each file was obtained. Do not assume redistribution is
> permitted until confirmed — some tweet-derived datasets license only tweet **IDs**,
> not tweet **text**.

- **`1k_data_emoji_tweets_senti_posneg.csv`**
  - Source: _to confirm_ (likely a public Kaggle / academic emoji-sentiment set)
  - License / terms: _to confirm_
  - Redistribution of text permitted: _to confirm_
- **`15_emoticon_data.csv`**
  - Source: _to confirm_ (emoji metadata; codepoints/names are factual Unicode data)
  - License / terms: _to confirm_

If an upstream license forbids redistributing the text, replace the committed CSV
with a fetch script (mirroring how the trained `.pkl` artifacts are pulled from GCS
at build time) and document the retrieval steps here instead.

## Privacy

The bundled corpora contain free-text social posts with sentiment labels and **no
direct identifiers**. Free text can still carry incidental personal references (an
`@mention` or a name inside a post); given the small size and public, label-only
nature of these sets the residual risk is low. Note that "publicly available" is not,
by itself, a legal basis for re-processing under GDPR/CCPA — attribution and respect
for the source license are what keep redistribution clean.
