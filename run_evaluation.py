import os
import sys
from src.pipeline import EvaluationPipeline

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    rubric_path = os.path.join(base_dir, "sample_data", "rubric.json")
    
    page1 = os.path.join(base_dir, "sample_data", "answer_sheet_page1.png")
    page2 = os.path.join(base_dir, "sample_data", "answer_sheet_page2.png")
    
    if not os.path.exists(page1) or not os.path.exists(page2):
        print("[Error] Sample dataset not found. Running dataset generator first...")
        import generate_sample_dataset
        generate_sample_dataset.generate_all()

    page_paths = [page1, page2]
    output_dir = os.path.join(base_dir, "output")

    pipeline = EvaluationPipeline(rubric_path=rubric_path)
    json_path, csv_path, results = pipeline.process_answer_sheets(page_paths, output_dir)

    print("\n=======================================================")
    print("      FINAL ANSWER SHEET EVALUATION SUMMARY            ")
    print("=======================================================")
    for res in results:
        print(f"\nQuestion: {res['question_no']} - {res['question_title']}")
        print(f"  Extracted Text Snippet: {res['extracted_answer_text'][:80]}...")
        print(f"  Score: {res['score_vs_rubric']}")
        print(f"  Confidence: [{res['confidence_label']}] Score: {res['confidence_score']}")
        print(f"  Reason: {res['confidence_reason']}")
        print("  Rubric Breakdown:")
        for crit in res['rubric_breakdown']:
            print(f"    - [{crit['criterion_id']}] {crit['description'][:50]}... => {crit['points_awarded']}/{crit['max_points']} (Sim: {crit.get('similarity_score', 0)})")

    print("\nDeliverables generated successfully:")
    print(f" - JSON Output: {json_path}")
    print(f" - CSV Output:  {csv_path}")

if __name__ == "__main__":
    main()
