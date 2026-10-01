import pytest
from src.layout_segmenter import LayoutSegmenter

def test_numbered_list_lines_are_not_headers():
    segmenter = LayoutSegmenter()
    elements_page1 = [
        {"page": 1, "bbox": [35, 180, 50, 30], "text": "Q1.", "is_margin": True, "is_crossed_out": False},
        {"page": 1, "bbox": [170, 180, 400, 30], "text": "Ans 1: TCP Handshake", "is_margin": False, "is_crossed_out": False},
        {"page": 1, "bbox": [170, 220, 800, 30], "text": "1. SYN: Client sends SYN segment", "is_margin": False, "is_crossed_out": False},
        {"page": 1, "bbox": [170, 260, 800, 30], "text": "2. SYN-ACK: Server responds with SYN-ACK", "is_margin": False, "is_crossed_out": False},
        {"page": 1, "bbox": [170, 300, 800, 30], "text": "3. ACK: Client confirms connection", "is_margin": False, "is_crossed_out": False},
    ]
    structured, _ = segmenter.segment_answers([elements_page1])
    
    assert len(structured) == 1
    assert structured[0]["question_no"] == "Q1"
    assert "1. SYN:" in structured[0]["extracted_answer_text"]
    assert "2. SYN-ACK:" in structured[0]["extracted_answer_text"]
    assert "3. ACK:" in structured[0]["extracted_answer_text"]

def test_q3_stitched_across_two_pages():
    segmenter = LayoutSegmenter()
    page1 = [
        {"page": 1, "bbox": [35, 680, 50, 30], "text": "Q3.", "is_margin": True, "is_crossed_out": False},
        {"page": 1, "bbox": [170, 680, 500, 30], "text": "Ans 3: OSI Reference Model", "is_margin": False, "is_crossed_out": False},
        {"page": 1, "bbox": [170, 720, 800, 30], "text": "OSI model has 7 layers.", "is_margin": False, "is_crossed_out": False},
        {"page": 1, "bbox": [170, 1000, 800, 30], "text": "[Continued on Page 2]", "is_margin": False, "is_crossed_out": False},
    ]
    page2 = [
        {"page": 2, "bbox": [35, 180, 50, 30], "text": "Q3.", "is_margin": True, "is_crossed_out": False},
        {"page": 2, "bbox": [170, 180, 600, 30], "text": "Ans 3 (Continued from Page 1): Key Layer Functions", "is_margin": False, "is_crossed_out": False},
        {"page": 2, "bbox": [170, 220, 800, 30], "text": "Detailed functions of primary OSI layers:", "is_margin": False, "is_crossed_out": False},
    ]
    structured, _ = segmenter.segment_answers([page1, page2])
    
    q3 = next(q for q in structured if q["question_no"] == "Q3")
    assert q3["pages"] == [1, 2]
    assert q3["is_multipage_spill"] is True
    assert "OSI model has 7 layers." in q3["extracted_answer_text"]
    assert "Detailed functions of primary OSI layers:" in q3["extracted_answer_text"]

def test_margin_note_attached_only_to_q4():
    segmenter = LayoutSegmenter()
    page2 = [
        {"page": 2, "bbox": [35, 180, 50, 30], "text": "Q3.", "is_margin": True, "is_crossed_out": False},
        {"page": 2, "bbox": [170, 180, 500, 30], "text": "Ans 3 (Continued): Key Layer Functions", "is_margin": False, "is_crossed_out": False},
        {"page": 2, "bbox": [35, 390, 50, 30], "text": "Q4.", "is_margin": True, "is_crossed_out": False},
        {"page": 2, "bbox": [170, 390, 500, 30], "text": "Ans 4: DNS Resolution Mechanism", "is_margin": False, "is_crossed_out": False},
        {"page": 2, "bbox": [15, 410, 100, 50], "text": "+1 Mark Bonus Check RFC 1035", "is_margin": True, "is_crossed_out": False},
    ]
    structured, _ = segmenter.segment_answers([page2])
    
    q3 = next(q for q in structured if q["question_no"] == "Q3")
    q4 = next(q for q in structured if q["question_no"] == "Q4")
    
    assert "+1 Mark Bonus Check RFC 1035" not in q3["margin_notes"]
    assert "+1 Mark Bonus Check RFC 1035" in q4["margin_notes"]

def test_crossed_out_line_excluded_from_scored_text():
    segmenter = LayoutSegmenter()
    page2 = [
        {"page": 2, "bbox": [35, 390, 50, 30], "text": "Q4.", "is_margin": True, "is_crossed_out": False},
        {"page": 2, "bbox": [170, 390, 500, 30], "text": "Ans 4: DNS Resolution Mechanism", "is_margin": False, "is_crossed_out": False},
        {"page": 2, "bbox": [170, 430, 800, 30], "text": "DNS translates hostnames to IP addresses.", "is_margin": False, "is_crossed_out": False},
        {"page": 2, "bbox": [170, 470, 800, 30], "text": "UDP is never used for DNS queries.", "is_margin": False, "is_crossed_out": True},
    ]
    structured, _ = segmenter.segment_answers([page2])
    
    q4 = next(q for q in structured if q["question_no"] == "Q4")
    assert "UDP is never used for DNS queries." not in q4["extracted_answer_text"]
    assert "UDP is never used for DNS queries." in q4["crossed_out_content"]
