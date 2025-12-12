from sentence_transformers import CrossEncoder

class HallucinationEvaluator:
    def __init__(self):
        # used a Cross-Encoder trained on NLI 
        self.model = CrossEncoder('cross-encoder/nli-deberta-v3-small')
        self.label_mapping = ['Hallucination', 'Faithful', 'Neutral']

    def evaluate(self, response: str, context: str) -> str:
        """
        determines if the response is supported by the context.
        """
        if not context:
            return "no context available"

        # limit context length to avoid token limits
        scores = self.model.predict([(context[:2000], response)])
        prediction_index = scores.argmax()
        
        return self.label_mapping[prediction_index]