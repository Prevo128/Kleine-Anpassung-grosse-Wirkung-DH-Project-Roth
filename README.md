# Kleine Anpassung, große Wirkung?

QLoRA-Finetuning für Named Entity Recognition auf historischen deutschen
Textkorpora (HIPE-2022), mit Energie-Tracking. Code zur gleichnamigen
Hausarbeit, Digital Humanities, Universität Leipzig.

## Worum es geht

Kann ein kleines LLM (Gemma 4 E2B) durch QLoRA-Finetuning auf unter 1%
seiner Parameter historische Named Entities zuverlässig erkennen, und wie
viel Energie kostet das im Verhältnis zum Leistungsgewinn? Getestet auf drei
deutschen HIPE-2022-Bundles (ajmc-de, hipe2020-de, newseye-de), jeweils
Zero-Shot-Baseline vs. Finetuned, auf einer einzelnen Consumer-GPU
(RTX 4070 Ti, 12GB VRAM).

## Struktur

```
pipeline.py              Einstiegspunkt: --prepare / --train / --evaluate
summarize_energy.py      Fasst outputs/emissions/emissions_log.csv zusammen
config/settings.yaml     Zentrale Konfiguration
src/
  energy_tracker.py       Energie-/Emissions-Tracker (codecarbon-Wrapper)
  train_reference.py      Trainingsskript (QLoRA-Finetuning)
  data_prep/
    convert_hipe.py        raw TSV -> cleaned prompt/completion JSON
    iob_utils.py            IOB2 <-> Entity-Tupel
  evaluation/
    run_inference.py        Modell laufen lassen, Output zurück nach IOB2/TSV
    run_scorer.py            ruft den offiziellen HIPE-Scorer auf
```

`raw/` (Originaldaten), `cleaned/` und `outputs/` (models/predictions/emissions)
werden nicht mitversioniert, siehe unten.

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Für Training/Inferenz zusätzlich:

```bash
pip install unsloth trl peft bitsandbytes torch transformers --break-system-packages
```

### Daten

HIPE-2022-Daten von [hipe-eval/HIPE-2022-data](https://github.com/hipe-eval/HIPE-2022-data)
laden, Original-Dateinamen können bleiben:

```
raw/
  ajmc-de/
    HIPE-2022-v2.1-ajmc-train-de.tsv
    HIPE-2022-v2.1-ajmc-dev-de.tsv
    HIPE-2022-v2.1-ajmc-test-de.tsv
  hipe2020-de/ ...
  newseye-de/ ...
```

Offizieller Scorer:

```bash
git clone https://github.com/hipe-eval/HIPE-scorer external/HIPE-scorer
pip install -r external/HIPE-scorer/requirements.txt
```

## Nutzung

```bash
python pipeline.py --prepare    # Daten aufbereiten + Baseline-Evaluation
python pipeline.py --train      # QLoRA-Finetuning
python pipeline.py --evaluate   # Finetuned-Evaluation + Scoring
python pipeline.py --train --bundle ajmc-de   # optional nur ein Bundle
python summarize_energy.py
```

Training direkt:

```bash
python -m src.train_reference --config config/settings.yaml --bundle ajmc-de
```

## Ergebnisse

Strict-F1, Zero-Shot-Baseline vs. Finetuned:

| Bundle | Baseline | Finetuned | Δ |
|---|---|---|---|
| ajmc-de | 1,0% | 83,0% | +82,0 |
| hipe2020-de | 1,9% | 66,1% | +64,2 |
| newseye-de | 0,0% | 39,6% | +39,6 |

Gesamtenergieverbrauch über die komplette Pipeline, alle drei Bundles
zusammen: rund 1391 Wh.

Details und Fehleranalyse: siehe Hausarbeit.
