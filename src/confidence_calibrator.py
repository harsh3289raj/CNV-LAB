class ConfidenceCalibrator:
    def __init__(self, ocr_threshold=0.75):
        self.ocr_threshold = ocr_threshold

    def calibrate_confidence(self, structured_answer, score, max_marks, criteria_breakdown, ocr_confidence=0.90):
        """
        Calibrates confidence level (HIGH, MEDIUM, LOW) and outputs a detailed one-line reason.
        """
        flags = []
        confidence_score = 1.0

        q_no = structured_answer.get("question_no")
        text = structured_answer.get("extracted_answer_text", "")
        crossed_out = structured_answer.get("crossed_out_content", "")
        margin_notes = structured_answer.get("margin_notes", "")
        has_diagram = structured_answer.get("has_diagram", False)
        is_multipage = structured_answer.get("is_multipage_spill", False)
        has_explicit_cont = "continued" in text.lower() or "contd" in text.lower()

        # Flag 1: Strikethrough / Crossed-out text present
        if crossed_out:
            confidence_score -= 0.35
            flags.append(f"Strikethrough text detected: '{crossed_out[:40]}...'")

        # Flag 2: Multi-page spill continuation
        if is_multipage:
            # Reduced spill penalty (0.05 vs 0.15) when explicit 'Continued' marker is present
            if has_explicit_cont:
                confidence_score -= 0.05
                flags.append("Answer spills across multiple pages (explicit continuation marker verified)")
            else:
                confidence_score -= 0.15
                flags.append("Answer spills across multiple pages (cross-page continuation stitched)")

        # Flag 3: Diagram check for Q3
        if q_no == "Q3" and not has_diagram:
            confidence_score -= 0.40
            flags.append("OSI diagram missing or unparsed")

        # Flag 4: Margin notes detected
        if margin_notes:
            confidence_score -= 0.10
            flags.append(f"Margin note present: '{margin_notes[:35]}...'")

        # Flag 5: Real OCR confidence evaluation
        if ocr_confidence < self.ocr_threshold:
            confidence_score -= 0.25
            flags.append(f"Low OCR legibility score ({ocr_confidence:.3f})")

        # Flag 6: Borderline semantic similarity / partial credit criterion
        partial_criteria = [c for c in criteria_breakdown if 0.35 <= c.get("similarity_score", 1.0) < 0.70]
        if partial_criteria:
            crit_ids = ", ".join(c.get("criterion_id", "?") for c in partial_criteria)
            confidence_score -= 0.15
            flags.append(f"Ambiguous semantic match on criteria [{crit_ids}] requiring human verification")

        # Include OCR score context in reason line
        ocr_info = f"[EasyOCR Conf: {ocr_confidence:.3f}]"

        # Determine label and concise one-line reason
        if confidence_score >= 0.80:
            confidence_label = "HIGH"
            reason = f"{ocr_info} High OCR clarity & clear rubric alignment." + (f" Note: {' | '.join(flags)}" if flags else "")
        elif confidence_score >= 0.55:
            confidence_label = "MEDIUM"
            reason = f"{ocr_info} Passable answer clarity; " + " | ".join(flags)
        else:
            confidence_label = "LOW"
            reason = f"{ocr_info} Requires human double-check: " + " | ".join(flags)

        return confidence_label, round(max(0.0, confidence_score), 2), reason

