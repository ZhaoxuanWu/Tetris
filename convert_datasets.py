import json
import os
from datasets import load_dataset

def obtain_arena_dataset(output_path = "Arena.json"):
    from datasets import load_dataset

    ds = load_dataset("lmsys/chatbot_arena_conversations")

    chats = []

    for idx, item in enumerate(ds['train']):
        conv = item['conversation_a']

        question = conv[0]['content']
        answer = conv[1]['content']

        chats.append({
            'id': idx,
            'conversations': [
                {
                    "from": "human",
                    "value": question
                },
                {
                    "from": "gpt",
                    "value": answer
                }
            ]
        })
    
    with open(output_path, "w") as f:
        json.dump(chats, f)

def obtain_domain_tough_dataset(output_path ="domain_tough.json"):

    ds = load_dataset("YAV-AI/llm-domain-specific-tough-questions")
    
    chats = []
    
    for idx, item in enumerate(ds["train"]):
        chats.append({
            'id': idx,
            'conversations': [
                {
                    "from": "human",
                    "value": item["question"]
                }
            ]
        })

    with open(output_path, "w") as f:
        json.dump(chats, f)


if __name__ == "__main__":
    obtain_arena_dataset()
    obtain_domain_tough_dataset()