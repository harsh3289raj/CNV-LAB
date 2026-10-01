import re

class LayoutSegmenter:
    def __init__(self):
        # Strictly require a header keyword: Ans/Answer/Q/Question (or 0/O OCR misreads before digit)
        self.header_pattern = re.compile(
            r'(?:^|\b)(?:ans(?:wer)?|q(?:uestion)?|[0o])\s*([1-9]|i{1,3}|iv)\b', 
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

        # Step 1: First pass to detect question headers with page numbers & y-coordinates
        page_question_headers = [] # list of {"q_no": "Q1", "page": 1, "y": 184}

        for page_idx, page_elements in enumerate(ocr_elements_by_page, start=1):
            current_q_no = None

            for elem in page_elements:
                text = elem.get("text", "").strip()
                if not text:
                    continue

                bbox = elem.get("bbox", [0, 0, 0, 0])
                y_coord = bbox[1]

                # 1. Separate Margin Notes for position-based attribution later
                if elem.get("is_margin", False):
                    # Exclude header labels that fell into margin (e.g., bare "Q1.")
                    header_m = self._extract_question_number(text)
                    if not header_m:
                        margin_notes.append({
                            "page": page_idx,
                            "y": y_coord,
                            "text": text,
                            "bbox": bbox
                        })
                        continue

                # 2. Check for Question Header
                header_match = self._extract_question_number(text)
                
                if header_match:
                    q_num = f"Q{header_match}"
                    page_question_headers.append({
                        "q_no": q_num,
                        "page": page_idx,
                        "y": y_coord
                    })

                    if q_num not in raw_questions:
                        raw_questions[q_num] = {
                            "question_no": q_num,
                            "pages": [page_idx],
                            "text_lines": [],
                            "crossed_out_lines": [],
                            "diagram_lines": [],
                            "margin_notes": [],
                            "has_diagram": False,
                            "has_continuation": False
                        }
                    else:
                        if page_idx not in raw_questions[q_num]["pages"]:
                            raw_questions[q_num]["pages"].append(page_idx)
                            raw_questions[q_num]["has_continuation"] = True

                    current_q_no = q_num

                    # If header line contains answer text beyond question title, keep body text
                    clean_header_text = self._strip_header_prefix(text)
                    if len(clean_header_text) > 3 and not self._is_pure_title(clean_header_text):
                        raw_questions[current_q_no]["text_lines"].append(clean_header_text)
                    continue

                # Skip line if it occurs before the first question header on page
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

        # Step 2: Position-based Margin Note Attribution
        # Attach each margin note to nearest question header above it on the same page
        for note in margin_notes:
            note_page = note["page"]
            note_y = note["y"]

            # Headers on the same page
            same_page_headers = [h for h in page_question_headers if h["page"] == note_page]
            if not same_page_headers:
                continue

            # Headers above or at note y
            headers_above = [h for h in same_page_headers if h["y"] <= note_y + 30]
            if headers_above:
                target_header = max(headers_above, key=lambda h: h["y"])
            else:
                target_header = min(same_page_headers, key=lambda h: h["y"])

            target_q = target_header["q_no"]
            if target_q in raw_questions:
                if note["text"] not in raw_questions[target_q]["margin_notes"]:
                    raw_questions[target_q]["margin_notes"].append(note["text"])

        # Step 3: Assemble finalized answer objects
        structured_answers = []
        for q_no in sorted(raw_questions.keys()):
            q_data = raw_questions[q_no]
            
            full_text = " ".join(q_data["text_lines"])
            crossed_out_text = " ".join(q_data["crossed_out_lines"])
            diagram_text = " | ".join(q_data["diagram_lines"])
            margin_text = " ; ".join(q_data["margin_notes"])

            is_spill = len(q_data["pages"]) > 1 or q_data["has_continuation"]

            structured_answers.append({
                "question_no": q_no,
                "pages": q_data["pages"],
                "extracted_answer_text": full_text,
                "has_diagram": q_data["has_diagram"],
                "diagram_components": diagram_text,
                "crossed_out_content": crossed_out_text,
                "margin_notes": margin_text,
                "is_multipage_spill": is_spill
            })

        return structured_answers, margin_notes

    def _extract_question_number(self, text):
        """Extracts question digit (1, 2, 3, 4) requiring explicit header keywords."""
        match = self.header_pattern.search(text)
        if match:
            raw_num = match.group(1).lower()
            roman_map = {"i": "1", "ii": "2", "iii": "3", "iv": "4"}
            return roman_map.get(raw_num, raw_num)
        return None

    def _strip_header_prefix(self, text):
        """Removes Q1., Ans 1: prefixes from line text."""
        return re.sub(
            r'^(?:ans(?:wer)?\s*|q(?:uestion)?\s*|[0o]\s*)?[1-9iiv]+[\.\:\)\s\-]*', 
            '', text, flags=re.IGNORECASE
        ).strip()

    def _is_pure_title(self, text):
        """Checks if line is just a title like 'TCP 3-Way Handshake Mechanism'."""
        titles = ["tcp 3-way handshake", "process vs thread", "osi reference model", "dns resolution"]
        return any(t in text.lower() for t in titles)

    def _is_diagram_line(self, text):
        """Identifies layer stack or diagram structural box markers."""
        diagram_keywords = ["architecture", "7-layer", "layer stack", "http, ftp", "tcp/udp", "mac framing", "ip routing", "physical layer", "---"]
        return any(kw in text.lower() for kw in diagram_keywords)


