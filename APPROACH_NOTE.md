# Approach Note: Handwritten Answer Sheet Evaluation Pipeline

## Architecture Overview
The pipeline processes multi-page handwritten exam answer sheets through a 4-stage architecture:
1. **Extract (EasyOCR):** Uses EasyOCR to detect text regions, groups word bounding boxes into spatial lines, extracts ROI features, and runs HSV color thresholding for strikethrough line detection.
2. **Structure (Layout Segmenter):** Uses explicit question header keyword matching (`Q1`, `Ans 2`), stitches multi-page answer continuations, and assigns margin notes to the nearest header above on the page.
3. **Score (Semantic Scorer):** Computes sentence embedding cosine similarity (`all-MiniLM-L6-v2`) between student answer sentences and fractional rubric criteria descriptions.
4. **Flag Confidence (Calibrator):** Computes risk penalties based on OCR legibility scores, strikethrough detection, diagram presence, cross-page stitching, and partial semantic matching.

## System Weaknesses & Honest Limitations
- **Synthetic Data Benchmark Only:** Pipeline and tests were evaluated against synthetic answer sheets rendered using Windows handwriting fonts. Performance on real cursive or messy student handwriting is unverified.
- **OCR Accuracy Limitations:** EasyOCR achieved a Word Error Rate (WER) of 0.4086 (59.14% word accuracy) on the synthetic handwriting sample, creating spelling artifacts in extracted text.
- **Heuristic Thresholds:** Embedding similarity cutoff thresholds (0.70 / 0.50 / 0.35) and confidence penalty values were chosen manually rather than calibrated against human grading datasets.
- **Basic Diagram Verification:** Diagram evaluation relies on simple keyword presence inside diagram text boxes without structural or multi-modal visual graph validation.
- **No Logical Reasoning Validation:** Sentence embedding similarity measures textual topic overlap but cannot verify mathematical correctness or logical validity.
