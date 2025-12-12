import json
from typing import Tuple, List

def load_json(filepath: str) -> dict:
    """Safely loads a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return {}

def extract_evaluation_data(chat_json: dict, context_json: dict) -> Tuple[str, str, str]:
    """
    Extracts (User Query, AI Response, Context) from the specific JSON formats provided.
    """
    # 1. Extract Context (Ground Truth)
    context_chunks = []
    vector_data = context_json.get('data', {}).get('vector_data', [])
    for item in vector_data:
        if 'text' in item:
            context_chunks.append(item['text'])
    context_text = " ".join(context_chunks)

    # 2. Extract AI Response
    sources = context_json.get('data', {}).get('sources', {})
    if 'final_response' in sources:
        response_text = " ".join(sources['final_response'])
    else:
        # Fallback: Get the very last message from the chat history
        turns = chat_json.get('conversation_turns', [])
        if turns and turns[-1]['role'] in ['AI/Chatbot', 'assistant']:
            response_text = turns[-1]['message']
        else:
            response_text = ""

    # 3. Extract User Query
    query_text = ""
    turns = chat_json.get('conversation_turns', [])
    # Iterate backwards to find the last user message
    for turn in reversed(turns):
        if turn['role'] in ['User', 'user']:
            query_text = turn['message']
            break

    return query_text, response_text, context_text