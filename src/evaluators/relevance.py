from sentence_transformers import SentenceTransformer, util

class RelevanceEvaluator:
    def __init__(self):
        # Load a lightweight embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def evaluate(self, query: str, response: str) -> float:
        """
        Returns a score 0-1 indicating how semantically close the response is to the query.
        """
        embeddings = self.model.encode([query, response])
        score = util.cos_sim(embeddings[0], embeddings[1])
        return float(score[0][0])