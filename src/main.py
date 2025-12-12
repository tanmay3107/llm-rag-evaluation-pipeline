import time
import json
import tiktoken
from utils import load_json, extract_evaluation_data
from evaluators.relevance import RelevanceEvaluator
from evaluators.hallucination import HallucinationEvaluator

class EvaluationPipeline:
    def __init__(self):
        print("Loading models... (it take a moment on first run)")
        self.relevance_eval = RelevanceEvaluator()
        self.hallucination_eval = HallucinationEvaluator()
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def calculate_cost(self, text: str) -> float:
        """estimates cost based on token count."""
        if not text: return 0.0
        tokens = len(self.encoding.encode(text))
        return (tokens / 1000) * 0.0015

    def run(self, chat_path: str, context_path: str):
        # 1. load Data
        chat_data = load_json(chat_path)
        context_data = load_json(context_path)
        
        # 2. extract Data
        query, response, context = extract_evaluation_data(chat_data, context_data)

        if not query or not response:
            print("Skipping: Could not find valid Query or Response.")
            return

        print(f"\n evaluation Report ")
        print(f"Query: {query[:100]}...")
        
        # 3. metrics
        start_time = time.time()
        cost = self.calculate_cost(response)
        relevance_score = self.relevance_eval.evaluate(query, response)
        
        # tiered evaluation logic
        if relevance_score < 0.2:
            hallucination_status = "Skipped (Irrelevant Response)"
        else:
            hallucination_status = self.hallucination_eval.evaluate(response, context)

        latency = time.time() - start_time

        # 4. output result
        result = {
            "metrics": {
                "latency_seconds": round(latency, 4),
                "estimated_cost_usd": round(cost, 6),
                "relevance_score": round(relevance_score, 3),
                "hallucination_status": hallucination_status
            },
            "data": {
                "query": query,
                "response_snippet": response[:100] + "...",
            }
        }
        
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    pipeline = EvaluationPipeline()
    print("Evaluating Conversation 1...")
    pipeline.run("data/sample-chat-conversation-01.json", "data/sample_context_vectors-01.json")