# Approach Note: Handwritten Answer Sheet Evaluation Pipeline

## Architecture Overview
The pipeline processes unformatted, real-world handwritten exam answer sheets through a 4-stage decoupled modular architecture:
1. **Extract (OCR/HTR Engine):** Employs spatial text region extraction with bounding-box geometry, line-level noise filtering, margin note segregation, and strikethrough detection.
2. **Structure (Layout Segmenter):** Uses header pattern recognition, cross-page continuation stitching (merging multi-page answers like Q3), and diagram region separation.
3. **Score (Semantic Scoring Engine):** Utilizes contextual sentence-transformers (`all-MiniLM-L6-v2`) cosine embeddings to grade student intent against fractional rubric criteria rather than brittle keyword matching.
4. **Flag Confidence (Calibrator & Risk Engine):** Computes risk scores based on OCR clarity, strikethrough lines, missing diagrams, cross-page stitching, and score boundary uncertainty to output human double-check flags.

## System Weaknesses & Edge Case Vulnerabilities
- **Messy & Overlapping Handwriting:** High cursive variance, faint ink strokes, or slanted handwriting can degrade standard OCR accuracy, resulting in OCR spelling artifacts.
- **Complex Diagram Understanding:** Current diagram handling relies on bounding box layout isolation and text extraction inside diagrams, but lacks deep multi-modal semantic validation of hand-drawn flowcharts/circuits.
- **Mathematical / Chemical Notation:** Non-textual mathematical equations, fraction bars, and chemical formulas are currently parsed as plain text rather than LaTeX syntax tree structures.
- **Implicit Question Transitions:** If a student omits explicit question labels (e.g. Q1, Q2) and relies only on spacing, layout segmentation relies on spatial layout clustering heuristics.

## Production Scaling & Future Improvements (What We'd Do With More Time)
- **Fine-Tuned Vision-Language Model (VLM):** Replace traditional OCR + NLP pipeline with a fine-tuned multimodal model (e.g., Qwen2-VL, Donut, Florence-2, or TrOCR) fine-tuned on real Indian exam board handwritten scripts.
- **Multimodal Diagram & Symbol Evaluation:** Integrate YOLO/LayoutLM diagram segmentation + Graph Neural Networks (GNNs) or SVG parsing for structural verification of diagrams, circuit schematics, and geometric figures.
- **Active Learning & Human-in-the-Loop Feedback:** Feed human evaluator double-check corrections back into a vector DB (RAG) to dynamically calibrate scoring thresholds across different subject evaluators.
- **Distributed Scale (Millions of Pages):** Implement async queue-based stream processing (Apache Kafka + Ray worker pool) with GPU batch inference to process 10,000+ pages/min during peak exam evaluation season.
