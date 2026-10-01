import os
import json
from src.ocr_engine import OCREngine

def levenshtein_distance(seq1, seq2):
    """Computes Levenshtein edit distance between two sequences of words."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]

def calculate_wer(reference_text, hypothesis_text):
    ref_words = reference_text.lower().split()
    hyp_words = hypothesis_text.lower().split()
    if not ref_words:
        return 0.0
    distance = levenshtein_distance(ref_words, hyp_words)
    return float(distance) / len(ref_words)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    gt_path = os.path.join(base_dir, "sample_data", "ground_truth.json")
    page1_path = os.path.join(base_dir, "sample_data", "answer_sheet_page1.png")
    page2_path = os.path.join(base_dir, "sample_data", "answer_sheet_page2.png")
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    with open(gt_path, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)

    ocr_engine = OCREngine(use_gpu=False)

    pages_eval = {}
    total_ref_words = 0
    total_edit_distance = 0

    page_paths = {"page1": page1_path, "page2": page2_path}

    for page_key, img_path in page_paths.items():
        page_num = 1 if page_key == "page1" else 2
        elements, _ = ocr_engine.extract_page_ocr(img_path, page_num=page_num)

        gt_lines = ground_truth.get(page_key, [])
        gt_full_text = " ".join(gt_lines)
        ocr_full_text = " ".join(e["text"] for e in elements)

        ref_words = gt_full_text.lower().split()
        hyp_words = ocr_full_text.lower().split()
        edit_dist = levenshtein_distance(ref_words, hyp_words)
        page_wer = float(edit_dist) / len(ref_words) if ref_words else 0.0

        total_ref_words += len(ref_words)
        total_edit_distance += edit_dist

        pages_eval[page_key] = {
            "ground_truth_text": gt_full_text,
            "ocr_extracted_text": ocr_full_text,
            "word_count_ground_truth": len(ref_words),
            "word_count_ocr": len(hyp_words),
            "edit_distance": edit_dist,
            "word_error_rate": round(page_wer, 4)
        }

    overall_wer = float(total_edit_distance) / total_ref_words if total_ref_words > 0 else 0.0

    eval_result = {
        "overall_word_error_rate": round(overall_wer, 4),
        "overall_accuracy_percentage": round((1.0 - overall_wer) * 100, 2),
        "total_reference_words": total_ref_words,
        "total_edit_distance": total_edit_distance,
        "pages": pages_eval
    }

    output_json_path = os.path.join(output_dir, "ocr_accuracy.json")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(eval_result, f, indent=2)

    print("\n=======================================================")
    print("           OCR ACCURACY EVALUATION RESULTS             ")
    print("=======================================================")
    print(f"Overall Word Error Rate (WER): {eval_result['overall_word_error_rate']} ({eval_result['overall_accuracy_percentage']}% Accuracy)")
    print(f"Page 1 WER: {pages_eval['page1']['word_error_rate']}")
    print(f"Page 2 WER: {pages_eval['page2']['word_error_rate']}")
    print(f"\nDeliverable saved to {output_json_path}")

    return eval_result

if __name__ == "__main__":
    main()
