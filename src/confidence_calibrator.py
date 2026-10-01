class ConfidenceCalibrator:
    def __init__(self, ocr_threshold=0.75):
        self.ocr_threshold = ocr_threshold

    def calibrate_confidence(self, structured_answer, score, max_marks, criteria_breakdown, ocr_confidence=0.90):
        """
        Calibrates confidence level (HIGH, MEDIUM, LOW) and outputs a concise one-line reason.
        """
        flags = []
        confidence_score = 1.0

        q_no = structured_answer.get("question_no")
        text = structured_answer.get("extracted_answer_text", "")
        crossed_out = structured_answer.get("crossed_out_content", "")
        margin_notes = structured_answer.get("margin_notes", "")
        has_diagram = structured_answer.get("has_diagram", False)
        is_multipage = structured_answer.get("is_multipage_spill", False)

        # Flag 1: Strikethrough / Crossed-out text present
        if crossed_out:
            confidence_score -= 0.35
            flags.append(f"Crossed-out text detected ('{crossed_out[:40]}...')")

        # Flag 2: Multi-page spill continuation
        if is_multipage:
            confidence_score -= 0.15
            flags.append("Answer spills across multiple pages (cross-page continuation stitched)")

        # Flag 3: Diagram check for Q3
        if q_no == "Q3" and not has_diagram:
            confidence_score -= 0.40
            flags.append("OSI diagram missing or unparsed")

        # Flag 4: Margin notes detected
        if margin_notes:
            confidence_score -= 0.10
            flags.append(f"Margin note present ('{margin_notes[:35]}...')")

        # Flag 5: OCR confidence
        if ocr_confidence < self.ocr_threshold:
            confidence_score -= 0.25
            flags.append(f"Low OCR legibility confidence ({ocr_confidence:.2f})")

        # Flag 6: Borderline semantic similarity / partial credit criterion
        partial_criteria = [c for c in criteria_breakdown if 0.35 <= c.get("similarity_score", 1.0) < 0.70]
        if partial_criteria:
            confidence_score -= 0.15
            flags.append(f"Ambiguous phrasing on {len(partial_criteria)} rubric criteria requiring semantic human verification")

        # Determine label and concise one-line reason
        if confidence_score >= 0.80:
            confidence_label = "HIGH"
            reason = "High OCR clarity, full rubric match, and clear un-ambiguous answer structure."
        elif confidence_score >= 0.55:
            confidence_label = "MEDIUM"
            reason = "Passable answer clarity; " + " | ".join(flags)
        else:
            confidence_label = "LOW"
            reason = "Requires human double-check: " + " | ".join(flags)

        return confidence_label, round(max(0.0, confidence_score), 2), reason
