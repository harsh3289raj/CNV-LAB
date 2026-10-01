import numpy as np
from sentence_transformers import SentenceTransformer, util

class SemanticScorer:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print(f"[Semantic Scorer] Loading semantic embedding model '{model_name}'...")
        try:
            self.model = SentenceTransformer(model_name)
            print("[Semantic Scorer] Embedding model loaded successfully.")
        except Exception as e:
            print(f"[Semantic Scorer] Warning: Could not load transformer model ({e}). Fallback to TF-IDF vector similarity.")
            self.model = None

    def calculate_similarity(self, text_a, text_b):
        """Calculates semantic similarity (0.0 to 1.0) between two text snippets."""
        if not text_a or not text_b:
            return 0.0
        
        if self.model is not None:
            emb1 = self.model.encode(text_a, convert_to_tensor=True)
            emb2 = self.model.encode(text_b, convert_to_tensor=True)
            cosine_sim = util.cos_sim(emb1, emb2).item()
            return float(np.clip(cosine_sim, 0.0, 1.0))
        else:
            # Fallback TF-IDF cosine similarity
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer().fit_transform([text_a, text_b])
            vectors = vectorizer.toarray()
            sim = np.dot(vectors[0], vectors[1]) / (np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1]) + 1e-9)
            return float(sim)

    def score_answer(self, structured_answer, question_rubric):
        """
        Scores a single student answer against its question rubric criteria.
        Returns total_score, max_marks, and criterion_breakdown.
        """
        q_no = structured_answer["question_no"]
        extracted_text = structured_answer["extracted_answer_text"]
        has_diagram = structured_answer.get("has_diagram", False)
        diagram_text = structured_answer.get("diagram_components", "")

        total_awarded = 0.0
        max_marks = question_rubric.get("max_marks", 5.0)
        criteria_breakdown = []

        # Combine student text and diagram components for comprehensive scoring
        combined_response = f"{extracted_text} {diagram_text}".strip()

        if not combined_response:
            return 0.0, max_marks, [{
                "criterion_id": "c0",
                "description": "Empty response",
                "points_awarded": 0.0,
                "max_points": max_marks,
                "similarity": 0.0
            }]

        # Split response into sentences for fine-grained criterion matching
        sentences = [s.strip() for s in combined_response.split('.') if len(s.strip()) > 3]
        if not sentences:
            sentences = [combined_response]

        for crit in question_rubric.get("rubric_criteria", []):
            c_id = crit["criterion_id"]
            c_desc = crit["description"]
            c_points = float(crit["points"])

            # Check if this criterion requires a diagram (e.g., Q3 diagram criterion)
            if "diagram" in c_desc.lower() or "stack" in c_desc.lower():
                if has_diagram or "layer" in diagram_text.lower():
                    similarity = 0.95
                    awarded = c_points
                else:
                    similarity = 0.20
                    awarded = 0.0
            else:
                # Find maximum semantic similarity across student sentences against criterion description
                sim_scores = [self.calculate_similarity(s, c_desc) for s in sentences]
                similarity = max(sim_scores) if sim_scores else 0.0

                # Also compare against model answer snippet similarity boost
                model_ans = question_rubric.get("model_answer", "")
                if model_ans:
                    model_sim = self.calculate_similarity(combined_response, model_ans)
                    similarity = max(similarity, model_sim * 0.95)

                # Soft threshold mapping for fractional rubric grading
                if similarity >= 0.70:
                    awarded = c_points
                elif similarity >= 0.50:
                    awarded = round(c_points * 0.75, 2)
                elif similarity >= 0.35:
                    awarded = round(c_points * 0.50, 2)
                else:
                    awarded = 0.0

            total_awarded += awarded

            criteria_breakdown.append({
                "criterion_id": c_id,
                "description": c_desc,
                "points_awarded": float(awarded),
                "max_points": float(c_points),
                "similarity_score": round(float(similarity), 3)
            })

        final_score = round(min(total_awarded, max_marks), 2)
        return final_score, max_marks, criteria_breakdown
