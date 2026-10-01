import re

class LayoutSegmenter:
    def __init__(self):
        self.header_pattern = re.compile(
            r'^(?:ans(?:wer)?\s*|q(?:uestion)?\s*)?([1-9])[\.\:\)\s\-]', 
            re.IGNORECASE
        )
        self.continuation_pattern = re.compile(
            r'(?:continued|contd|cont\.?\s*from\s*page)', 
            re.IGNORECASE
        )

    def segment_answers(self, ocr_elements_by_page):
        """
        Segments raw OCR text elements across pages into structured question objects.
        Handles cross-page continuation, margin notes, strikethrough lines, and diagrams.
        """
        raw_questions = {}
        margin_notes = []

        for page_idx, page_elements in enumerate(ocr_elements_by_page, start=1):
            current_q_no = None

            for elem in page_elements:
                text = elem.get("text", "").strip()
                if not text:
                    continue

                # 1. Handle Margin Notes
                if elem.get("is_margin", False):
                    margin_notes.append({
                        "page": page_idx,
                        "text": text,
                        "bbox": elem.get("bbox")
                    })
                    continue

                # 2. Check for Question Header (e.g., "Q1.", "Ans 1: TCP Handshake", "Ans 3 (Continued)")
                header_match = self._extract_question_number(text)
                
                if header_match:
                    q_num = f"Q{header_match}"
                    if q_num not in raw_questions:
                        raw_questions[q_num] = {
                            "question_no": q_num,
                            "pages": [page_idx],
                            "text_lines": [],
                            "crossed_out_lines": [],
                            "diagram_lines": [],
                            "has_diagram": False,
                            "has_continuation": False
                        }
                    else:
                        if page_idx not in raw_questions[q_num]["pages"]:
                            raw_questions[q_num]["pages"].append(page_idx)
                            raw_questions[q_num]["has_continuation"] = True

                    current_q_no = q_num

                    # If header line contains answer text beyond title, keep text
                    clean_header_text = self._strip_header_prefix(text)
                    if len(clean_header_text) > 3 and not self._is_pure_title(clean_header_text):
                        raw_questions[current_q_no]["text_lines"].append(clean_header_text)
                    continue

                # Skip if line occurs before first question header on page
                if current_q_no is None:
                    continue

                # 3. Handle Strikethrough / Crossed-out lines
                if elem.get("is_crossed_out", False) or text.startswith("~~"):
                    clean_txt = text.replace("~~", "").strip()
                    raw_questions[current_q_no]["crossed_out_lines"].append(clean_txt)
                    continue

                # 4. Handle Diagram Lines & Continuation markers
                if self.continuation_pattern.search(text):
                    raw_questions[current_q_no]["has_continuation"] = True
                    continue

                if self._is_diagram_line(text):
                    raw_questions[current_q_no]["has_diagram"] = True
                    raw_questions[current_q_no]["diagram_lines"].append(text)
                else:
                    raw_questions[current_q_no]["text_lines"].append(text)

        # Assemble finalized answer strings per question
        structured_answers = []
        for q_no in sorted(raw_questions.keys()):
            q_data = raw_questions[q_no]
            
            full_text = " ".join(q_data["text_lines"])
            crossed_out_text = " ".join(q_data["crossed_out_lines"])
            diagram_text = " | ".join(q_data["diagram_lines"])

            # Find margin notes associated ONLY with this question's pages
            q_margins = [m["text"] for m in margin_notes if m["page"] in q_data["pages"] and not m["text"].startswith("Q")]

            structured_answers.append({
                "question_no": q_no,
                "pages": q_data["pages"],
                "extracted_answer_text": full_text,
                "has_diagram": q_data["has_diagram"],
                "diagram_components": diagram_text,
                "crossed_out_content": crossed_out_text,
                "margin_notes": " ; ".join(q_margins),
                "is_multipage_spill": q_data["has_continuation"] or len(q_data["pages"]) > 1
            })

        return structured_answers, margin_notes

    def _extract_question_number(self, text):
        """Extracts question digit (1, 2, 3, 4) from line header text."""
        match = re.search(r'(?:ans(?:wer)?\s*|q(?:uestion)?\s*|^)([1-9])\b', text, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def _strip_header_prefix(self, text):
        """Removes Q1., Ans 1: prefixes."""
        return re.sub(r'^(?:ans(?:wer)?\s*|q(?:uestion)?\s*)?[1-9][\.\:\)\s\-]*', '', text, flags=re.IGNORECASE).strip()

    def _is_pure_title(self, text):
        """Checks if line is just a title like 'TCP 3-Way Handshake Mechanism'."""
        titles = ["tcp 3-way handshake", "process vs thread", "osi reference model", "dns resolution"]
        return any(t in text.lower() for t in titles)

    def _is_diagram_line(self, text):
        """Identifies layer stack or diagram structural box markers."""
        diagram_keywords = ["architecture", "layer", "http, ftp", "tcp/udp", "mac framing", "ip routing", "physical layer", "---"]
        return any(kw in text.lower() for kw in diagram_keywords)
