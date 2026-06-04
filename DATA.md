# Data Card

## Overview

This document records the datasets bundled in this repository, their structure, and — importantly — their provenance and licensing, which are **separate from the project's code license**. The repository's `LICENSE` (MIT) covers the source code only; it does **not** grant any rights over the third-party data files under `src/data/`. The corpora here are small, label-only sentiment sets with no direct personal identifiers (no usernames, user IDs, timestamps, or geolocation), so the privacy surface is low, but redistribution terms depend on each set's original license. These files were provided as material for a school activity with no accompanying license, so their redistribution terms are unverified and sharing them should not be treated as cleared.

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

These files were **provided by the course instructor as material for a school
activity**. The original upstream origin was not recorded at the time, and no
license accompanied them, so their **redistribution terms are unverified**.

- **`1k_data_emoji_tweets_senti_posneg.csv`**
  - Source: provided as coursework material; upstream origin not recorded
  - License / terms: unverified
  - Redistribution of text permitted: unverified
- **`15_emoticon_data.csv`**
  - Source: provided as coursework material; upstream origin not recorded
  - License / terms: unverified (codepoints/names are factual Unicode data)

Because the provenance is a classroom hand-out rather than a licensed public
release, treat redistribution as **not cleared**. If the upstream terms ever
surface and forbid sharing the tweet text, replace the committed CSV with a fetch
script (mirroring how the trained `.pkl` artifacts are pulled from GCS at build
time) and document the retrieval steps here instead.

## Privacy

The bundled corpora contain free-text social posts with sentiment labels and **no
direct identifiers**. Free text can still carry incidental personal references (an
`@mention` or a name inside a post); given the small size and public, label-only
nature of these sets the residual risk is low. Note that "publicly available" is not,
by itself, a legal basis for re-processing under GDPR/CCPA — attribution and respect
for the source license are what keep redistribution clean.
