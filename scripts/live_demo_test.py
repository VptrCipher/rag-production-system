import json
import logging
from fastapi.testclient import TestClient
from api.main import app

# Suppress debug logs for cleaner output
logging.getLogger("httpx").setLevel(logging.WARNING)

client = TestClient(app)

queries = [
    {
        "name": "General Question",
        "question": "How does a vector database improve RAG performance compared to standard keyword search?"
    },
    {
        "name": "Buggy Python Code (IndexError)",
        "question": "I have this python code: \n\nmy_list = [1, 2, 3]\nfor i in range(4):\n    print(my_list[i])\n\nIt throws an IndexError. Can you explain why and provide the corrected code?"
    },
    {
        "name": "Buggy JavaScript Code (Syntax Error)",
        "question": "This javascript snippet isn't working: \n\nconst greet = (name) => {\n  console.log('Hello, ' + name\n}\n\nWhat is wrong with it and how do I fix it?"
    }
]

def run_live_test():
    results = []
    print("="*60)
    print("LIVE RAG MODEL QUERY DEMONSTRATION")
    print("="*60 + "\n")
    
    for i, q in enumerate(queries, 1):
        print(f"Test #{i}: {q['name']}")
        print(f"User Query:\n{q['question']}\n")
        
        print("Processing query through RAG pipeline...\n")
        response = client.post(
            "/api/v1/query",
            json={"question": q['question'], "session_id": "live_demo_test"}
        )
        
        result_entry = {
            "test_name": q['name'],
            "question": q['question'],
            "status_code": response.status_code
        }
        
        if response.status_code == 200:
            data = response.json()
            result_entry["answer"] = data.get('answer')
            result_entry["sources_count"] = len(data.get('sources', []))
            if "metadata" in data:
                result_entry["latency_ms"] = data['metadata'].get('latency_ms', 'N/A')
                result_entry["model"] = data['metadata'].get('model', 'N/A')
            print(f"Status: SUCCESS (200 OK)")
        else:
            result_entry["error"] = response.text
            print(f"Status: FAILED ({response.status_code})")
        
        results.append(result_entry)
        print("\n" + "-"*60 + "\n")
        
    with open("live_demo_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_live_test()
