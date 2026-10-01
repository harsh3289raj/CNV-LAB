import os
import json
import csv
import cv2
import numpy as np
from src.ocr_engine import OCREngine
from src.layout_segmenter import LayoutSegmenter
from src.semantic_scorer import SemanticScorer
from src.confidence_calibrator import ConfidenceCalibrator

class EvaluationPipeline:
    def __init__(self, rubric_path, model_name="all-MiniLM-L6-v2"):
        print("[Pipeline] Initializing AI/ML Answer Sheet Evaluation Pipeline...")
        
        # Load Rubric JSON
        with open(rubric_path, 'r', encoding='utf-8') as f:
            self.rubric = json.load(f)
            
        self.rubric_dict = {q["question_no"]: q for q in self.rubric["questions"]}
        
        # Initialize Step Modules
        self.ocr_engine = OCREngine()
        self.layout_segmenter = LayoutSegmenter()
        self.semantic_scorer = SemanticScorer(model_name=model_name)
        self.confidence_calibrator = ConfidenceCalibrator()

    def process_answer_sheets(self, page_paths, output_dir):
        """
        Executes 4-step pipeline:
        Step 1: Extract (OCR / Handwriting Recognition)
        Step 2: Structure (Segmentation, Cross-page continuation stitching, noise filtering)
        Step 3: Score (Semantic scoring against rubric criteria)
        Step 4: Flag Confidence (Confidence calibration & human review flagging)
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n--- STEP 1: EXTRACTING OCR TEXT FROM {len(page_paths)} PAGES ---")
        
        ocr_elements_by_page = []
        page_dimensions = []
        
        for idx, page_path in enumerate(page_paths, start=1):
            print(f" -> Processing Page {idx}: {page_path}")
            elements, dims = self.ocr_engine.extract_page_ocr(page_path, page_num=idx)
            ocr_elements_by_page.append(elements)
            page_dimensions.append(dims)

        print(f"\n--- STEP 2: STRUCTURING & SEGMENTING BY QUESTION NUMBER ---")
        structured_answers, margin_notes = self.layout_segmenter.segment_answers(ocr_elements_by_page)
        
        for ans in structured_answers:
            print(f" -> Found {ans['question_no']} (Pages: {ans['pages']}) | Diagram: {ans['has_diagram']} | Multi-page Spill: {ans['is_multipage_spill']}")

        print(f"\n--- STEP 3 & 4: SEMANTIC SCORING & CONFIDENCE CALIBRATION ---")
        results = []

        for ans in structured_answers:
            q_no = ans["question_no"]
            q_rubric = self.rubric_dict.get(q_no, {})
            
            exp_score = q_rubric.get("expected_score", None)
            if exp_score is None:
                print(f"[Warning] 'expected_score' for {q_no} is null in rubric.json. Please fill in expected scores from your answer key.")

            # Step 3: Score
            score, max_marks, criteria_breakdown = self.semantic_scorer.score_answer(ans, q_rubric)
            
            score_diff = round(score - exp_score, 2) if exp_score is not None else None

            # Compute average OCR confidence for answer's elements
            q_ocr_confs = []
            for p_idx in ans["pages"]:
                p_elems = ocr_elements_by_page[p_idx - 1]
                for e in p_elems:
                    q_ocr_confs.append(e.get("confidence", 0.90))
            avg_ocr_conf = float(np.mean(q_ocr_confs)) if q_ocr_confs else 0.90

            # Step 4: Flag Confidence
            conf_label, conf_score, conf_reason = self.confidence_calibrator.calibrate_confidence(
                ans, score, max_marks, criteria_breakdown, ocr_confidence=avg_ocr_conf
            )

            result_entry = {
                "question_no": q_no,
                "question_title": q_rubric.get("title", ""),
                "extracted_answer_text": ans["extracted_answer_text"],
                "has_diagram": ans["has_diagram"],
                "diagram_components": ans["diagram_components"],
                "crossed_out_content": ans["crossed_out_content"],
                "margin_notes": ans["margin_notes"],
                "spills_across_pages": ans["is_multipage_spill"],
                "score": score,
                "max_marks": max_marks,
                "expected_score": exp_score,
                "score_difference": score_diff,
                "score_vs_rubric": f"{score}/{max_marks}",
                "rubric_breakdown": criteria_breakdown,
                "confidence_label": conf_label,
                "confidence_score": conf_score,
                "confidence_reason": conf_reason
            }

            results.append(result_entry)
            print(f" -> [{q_no}] Score: {score}/{max_marks} (Expected: {exp_score}) | Confidence: {conf_label} | Reason: {conf_reason}")

        # Save JSON output
        json_output_path = os.path.join(output_dir, "evaluation_results.json")
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "subject": self.rubric.get("subject", ""),
                "total_score": round(sum(r["score"] for r in results), 2),
                "total_max_marks": self.rubric.get("total_marks", 20.0),
                "evaluated_questions": results,
                "detected_margin_notes": margin_notes
            }, f, indent=2)

        # Save CSV output
        csv_output_path = os.path.join(output_dir, "evaluation_results.csv")
        with open(csv_output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "question_no", "score", "max_marks", "expected_score", "score_difference",
                "score_vs_rubric", "confidence_label", "confidence_reason", 
                "extracted_answer_text", "has_diagram", "crossed_out_content", "spills_across_pages"
            ])
            for r in results:
                writer.writerow([
                    r["question_no"], r["score"], r["max_marks"], 
                    r["expected_score"] if r["expected_score"] is not None else "",
                    r["score_difference"] if r["score_difference"] is not None else "",
                    r["score_vs_rubric"], r["confidence_label"], r["confidence_reason"], 
                    r["extracted_answer_text"], r["has_diagram"], r["crossed_out_content"], r["spills_across_pages"]
                ])

        # Generate Visual Bounding Box Annotations
        self._generate_annotated_images(page_paths, ocr_elements_by_page, results, output_dir)

        print(f"\n[Pipeline] Completed successfully! Deliverables saved to {output_dir}")
        return json_output_path, csv_output_path, results

    def _generate_annotated_images(self, page_paths, ocr_elements_by_page, results, output_dir):
        """Generates annotated inspection images showing bounding boxes, question tags, and flags."""
        for idx, page_path in enumerate(page_paths, start=1):
            img = cv2.imread(page_path)
            if img is None:
                continue

            elems = ocr_elements_by_page[idx - 1]
            for e in elems:
                x, y, w, h = e["bbox"]
                if e.get("is_margin"):
                    color = (255, 100, 0) # Orange for margin notes
                    label = "MARGIN"
                elif e.get("is_crossed_out"):
                    color = (0, 0, 255) # Red for strikethrough
                    label = "STRIKETHROUGH"
                else:
                    color = (0, 180, 0) # Green for text
                    label = "TEXT"

                cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)

            out_img_path = os.path.join(output_dir, f"annotated_page{idx}.png")
            cv2.imwrite(out_img_path, img)
            print(f" -> Saved visual inspection overlay: {out_img_path}")
