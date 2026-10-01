# Answer Sheet Evaluation Pipeline

An automated AI/ML evaluation pipeline for reading, segmenting, scoring, and calibrating confidence on handwritten student answer sheets against predefined rubric criteria.

## Overview
This project provides an end-to-end Python processing pipeline that reads multi-page handwritten exam answer sheets using EasyOCR, extracts line-level spatial bounding boxes, segments answer blocks by question number, grades answer content against a fractional point rubric using sentence embedding similarity (`all-MiniLM-L6-v2`), and flags edge cases (strikethroughs, margin notes, cross-page spills) with confidence calibration labels.

## Pipeline Architecture

The processing pipeline is divided into four decoupled stages:

| Stage | Description | Module File |
| :--- | :--- | :--- |
| **1. Extract** | Preprocesses page images, detects text bounding boxes using live EasyOCR, groups word boxes into lines, filters noise, and detects strikethroughs via HSV color thresholding. | [`src/ocr_engine.py`](src/ocr_engine.py) |
| **2. Structure** | Parses question headers requiring explicit header keywords (`Ans`, `Q`), stitches multi-page answer continuations, assigns margin notes to the nearest header above on the page, and isolates diagram blocks. | [`src/layout_segmenter.py`](src/layout_segmenter.py) |
| **3. Score** | Computes cosine similarity between student text embeddings and rubric criteria descriptions using `sentence-transformers` (`all-MiniLM-L6-v2`). | [`src/semantic_scorer.py`](src/semantic_scorer.py) |
| **4. Flag Confidence** | Evaluates real OCR legibility scores, strikethroughs, diagram presence, cross-page continuation markers, and semantic ambiguity to assign a confidence rating (`HIGH`, `MEDIUM`, `LOW`). | [`src/confidence_calibrator.py`](src/confidence_calibrator.py) |

The complete sequence is orchestrated by [`src/pipeline.py`](src/pipeline.py).

## Recent Fixes & Improvements ("What Changed")

1. **Live EasyOCR Pipeline:** Replaced hardcoded fallback text parser with live EasyOCR image recognition (`download_enabled=True`). If OCR fails or detects no text, the pipeline raises a clear runtime exception rather than silently using placeholder text.
2. **Fixed Question Segmentation Bug:** Resolved a bug where bare line numbers (e.g. `1. SYN: ...`, `2. SYN-ACK: ...`) were misidentified as question headers. Header parsing now strictly requires explicit keywords (`Ans`, `Answer`, `Q`, `Question`).
3. **Correct Margin Note Attribution:** Margin notes are now attached to the single nearest question header above or at the note's vertical position on that page, rather than every question on the page.
4. **OCR WER Accuracy Evaluation:** Added [`evaluate_ocr.py`](evaluate_ocr.py) to benchmark OCR output against [`sample_data/ground_truth.json`](sample_data/ground_truth.json).
5. **Unit Test Suite:** Added pytest test suite in [`tests/test_segmenter.py`](tests/test_segmenter.py) verifying header segmentation, multi-page stitching, margin note attribution, and strikethrough exclusion.

## Folder Structure

```
.
├── APPROACH_NOTE.md           # Technical approach note and honest system limitations
├── README.md                  # Project documentation and run instructions
├── evaluate_ocr.py            # OCR Word Error Rate (WER) accuracy benchmark script
├── generate_sample_dataset.py # Script for generating synthetic handwritten answer sheets
├── requirements.txt           # Python package dependencies
├── run_evaluation.py          # Pipeline execution script
├── output/                    # Evaluation output deliverables
│   ├── annotated_page1.png    # Page 1 bounding box annotation overlay
│   ├── annotated_page2.png    # Page 2 bounding box annotation overlay
│   ├── evaluation_results.csv # Tabular CSV evaluation summary
│   ├── evaluation_results.json# Detailed JSON breakdown with per-criterion scores
│   └── ocr_accuracy.json     # EasyOCR WER accuracy benchmark results
├── sample_data/               # Input sample dataset
│   ├── answer_sheet_page1.png # Page 1 sample handwritten answer sheet image
│   ├── answer_sheet_page2.png # Page 2 sample handwritten answer sheet image
│   ├── ground_truth.json      # Ground truth text lines for OCR WER benchmarking
│   ├── rubric.json            # Grading rubric criteria (expected_score set to null)
│   └── source_metadata.json   # Source metadata for synthetic dataset
├── src/                       # Core python pipeline modules
│   ├── __init__.py
│   ├── confidence_calibrator.py
│   ├── layout_segmenter.py
│   ├── ocr_engine.py
│   ├── pipeline.py
│   └── semantic_scorer.py
└── tests/                     # Unit test suite
    ├── __init__.py
    └── test_segmenter.py
```

## Setup and How to Run

### Prerequisites
- Python 3.9+
- `pip` package manager

### Installation
Clone the repository and install the required Python packages:
```bash
pip install -r requirements.txt
```

### Execution

1. **Run Evaluation Pipeline:**
   ```bash
   python run_evaluation.py
   ```

2. **Run OCR Accuracy Benchmark:**
   ```bash
   python evaluate_ocr.py
   ```

3. **Run Unit Tests:**
   ```bash
   python -m pytest
   ```

## Output Files

Executing the pipeline populates the `output/` directory with:
- **`output/evaluation_results.json`**: Complete JSON output containing total score, expected scores (`null`), score difference, per-question score breakdown, similarity metrics, confidence labels, and confidence calibration reasons.
- **`output/evaluation_results.csv`**: CSV summary containing question scores, expected scores, score differences, confidence levels, and extracted text snippets.
- **`output/ocr_accuracy.json`**: Benchmark results comparing EasyOCR output against ground truth text, reporting WER per page and overall.
- **`output/annotated_page1.png` & `output/annotated_page2.png`**: Visual inspection overlays highlighting detected text blocks (green), margin notes (orange), and strikethrough lines (red).

## Evaluation Results

Evaluation summary on the synthetic 2-page sample dataset (Max Total Marks: 20.0):

| Question | Score | Expected Score | Confidence Label | Summary / Flag Reason |
| :--- | :---: | :---: | :---: | :--- |
| **Q1: TCP 3-Way Handshake** | **5.00 / 5.0** | *null* | `MEDIUM` | `[EasyOCR Conf: 0.702]` Passable answer clarity; Low OCR legibility score |
| **Q2: Process vs Thread** | **5.00 / 5.0** | *null* | `MEDIUM` | `[EasyOCR Conf: 0.702]` Passable answer clarity; Low OCR legibility score |
| **Q3: OSI Reference Model** | **5.00 / 5.0** | *null* | `MEDIUM` | `[EasyOCR Conf: 0.678]` Passable answer clarity; Explicit continuation marker verified; Low OCR legibility score |
| **Q4: DNS Resolution Mechanism** | **5.00 / 5.0** | *null* | `LOW` | `[EasyOCR Conf: 0.641]` Requires human double-check: Strikethrough text detected (`HDEis heYer_uSed...`); Margin note present (`+1 Mark Bonus ...`); Low OCR legibility score |

*Note: `expected_score` values are set to `null` in `sample_data/rubric.json` pending user answer key input.*

## OCR Accuracy Benchmark

Benchmarked using [`evaluate_ocr.py`](evaluate_ocr.py) against [`sample_data/ground_truth.json`](sample_data/ground_truth.json):

* **Overall Word Error Rate (WER):** `0.4086` (59.14% Word Accuracy)
* **Page 1 WER:** `0.3458`
* **Page 2 WER:** `0.5074`

## Source of Sample Data

The sample answer sheet images (`sample_data/answer_sheet_page1.png` and `sample_data/answer_sheet_page2.png`) are synthetic sample files generated programmatically by [`generate_sample_dataset.py`](generate_sample_dataset.py). The generator renders text using Windows handwriting fonts (Segoe Script, Ink Free) onto a procedurally generated paper texture background with ruled notebook lines, hand-drawn diagram boxes, red margin lines, and slight affine camera rotation.
