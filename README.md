# Answer Sheet Evaluation Pipeline

An automated AI/ML evaluation pipeline for scanning, segmenting, scoring, and calibrating confidence on handwritten student answer sheets against predefined rubric criteria.

## Overview
This project provides an end-to-end Python processing pipeline that reads multi-page handwritten exam answer sheets via OCR, extracts line-level spatial bounding boxes, segments answer blocks by question number, grades answer content against a fractional point rubric using sentence embedding similarity (`all-MiniLM-L6-v2`), and flags edge cases (strikethroughs, margin notes, cross-page spills) with confidence calibration labels.

## Pipeline Architecture

The processing pipeline is divided into four decoupled stages:

| Stage | Description | Module File |
| :--- | :--- | :--- |
| **1. Extract** | Preprocesses page images, extracts bounding box ROI regions, filters noise, flags strikethroughs, and reads text tokens. | [`src/ocr_engine.py`](file:///src/ocr_engine.py) |
| **2. Structure** | Parses question headers, stitches multi-page answer continuations (e.g., Q3 spanning pages 1 & 2), separates margin notes, and isolates diagram blocks. | [`src/layout_segmenter.py`](file:///src/layout_segmenter.py) |
| **3. Score** | Computes cosine similarity between student text embeddings and rubric criteria descriptions using `sentence-transformers`. | [`src/semantic_scorer.py`](file:///src/semantic_scorer.py) |
| **4. Flag Confidence** | Evaluates OCR legibility, strikethroughs, diagram presence, and semantic ambiguity to assign a confidence rating (`HIGH`, `MEDIUM`, `LOW`) with a detailed reason. | [`src/confidence_calibrator.py`](file:///src/confidence_calibrator.py) |

The complete sequence is orchestrated by [`src/pipeline.py`](file:///src/pipeline.py).

## Folder Structure

```
.
├── APPROACH_NOTE.md           # Detailed technical approach and architecture design note
├── README.md                  # Project documentation and run instructions
├── generate_sample_dataset.py # Script for generating synthetic handwritten answer sheets
├── requirements.txt           # Python package dependencies
├── run_evaluation.py          # Entry point execution script
├── output/                    # Evaluation output deliverables
│   ├── annotated_page1.png    # Page 1 spatial bounding box annotation overlay
│   ├── annotated_page2.png    # Page 2 spatial bounding box annotation overlay
│   ├── evaluation_results.csv # Evaluation summary in CSV format
│   └── evaluation_results.json# Detailed JSON breakdown with per-criterion scores
├── sample_data/               # Input sample data files
│   ├── answer_sheet_page1.png # Page 1 sample handwritten answer sheet image
│   ├── answer_sheet_page2.png # Page 2 sample handwritten answer sheet image
│   ├── rubric.json            # Grading rubric criteria and point distribution
│   └── source_metadata.json   # Source metadata for sample input
└── src/                       # Core python pipeline modules
    ├── __init__.py
    ├── confidence_calibrator.py
    ├── layout_segmenter.py
    ├── ocr_engine.py
    ├── pipeline.py
    └── semantic_scorer.py
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
Run the complete evaluation pipeline:
```bash
python run_evaluation.py
```

If the sample image files in `sample_data/` are missing, `run_evaluation.py` will automatically invoke `generate_sample_dataset.py` to recreate them before running the evaluation.

## Output Files

Executing the pipeline populates the `output/` directory with:
- **`output/evaluation_results.json`**: Complete JSON output containing total score, per-question score breakdown, similarity metrics, confidence labels, and confidence calibration reasons.
- **`output/evaluation_results.csv`**: Tabular CSV summary containing question scores, confidence levels, and extracted text snippets.
- **`output/annotated_page1.png` & `output/annotated_page2.png`**: Inspection image overlays highlighting detected text blocks (green), margin notes (orange), and strikethrough lines (red).

## Evaluation Results

Evaluation summary on the sample 2-page dataset (Max Total Marks: 20.0):

| Question | Score | Confidence Label | Summary / Flag Reason |
| :--- | :---: | :---: | :--- |
| **Q1: TCP 3-Way Handshake** | **5.00 / 5.0** | `MEDIUM` | Passable answer clarity; Answer spills across multiple pages; Margin note present |
| **Q2: Process vs Thread** | **4.24 / 5.0** | `MEDIUM` | Passable answer clarity; Answer spills across multiple pages; Margin note present; Ambiguous phrasing on 2 rubric criteria |
| **Q3: OSI Reference Model** | **4.00 / 5.0** | `MEDIUM` | Passable answer clarity; Answer spills across multiple pages; Margin note present; Ambiguous phrasing on 2 rubric criteria |
| **Q4: DNS Resolution Mechanism** | **3.26 / 5.0** | `LOW` | Requires human double-check: Crossed-out text detected (`UDP is never used...`); Margin note present; Ambiguous phrasing on 2 rubric criteria |

## Source of Sample Data

The two sample answer sheet images (`sample_data/answer_sheet_page1.png` and `sample_data/answer_sheet_page2.png`) are synthetic sample files generated programmatically by [`generate_sample_dataset.py`](file:///generate_sample_dataset.py). The generator renders text using natural handwriting TTF fonts (e.g. Segoe Script, Ink Free) onto a procedurally generated paper texture background with ruled notebook lines, hand-drawn diagrams, red margin lines, and slight affine camera rotation.

## Known Limitations and Next Steps

*(Adapted from [`APPROACH_NOTE.md`](file:///APPROACH_NOTE.md))*

### Known Limitations
- **Handwriting Variance:** Highly cursive, unaligned, or faint handwriting can degrade OCR detection, introducing character recognition artifacts.
- **Basic Diagram Parsing:** Current diagram handling relies on bounding box layout isolation and text extraction inside diagrams, but lacks deep multi-modal semantic validation of hand-drawn flowcharts or schematics.
- **Mathematical & Chemical Notation:** Equations, fraction bars, and chemical formulas are currently parsed as plain text lines rather than structured LaTeX syntax trees.
- **Implicit Transitions:** Answers without explicit question headers (e.g., `Q1`, `Ans 2`) rely on spatial clustering heuristics.

### Next Steps & Future Enhancements
- **Vision-Language Models (VLM):** Transitioning to multimodal vision-language models (e.g., Qwen2-VL, Donut, or TrOCR) fine-tuned on real handwritten exam scripts.
- **Multimodal Diagram & Symbol Evaluation:** Integrating GNNs or SVG parsing for structural verification of diagrams, circuit schematics, and geometric figures.
- **Active Learning & Feedback:** Feeding human evaluator double-check corrections back into a vector database (RAG) to dynamically calibrate scoring thresholds.
- **Distributed Scale:** Implementing async queue processing (Apache Kafka + Ray worker pool) with GPU batch inference to process large volumes of pages efficiently.
