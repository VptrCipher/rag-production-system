import requests
import json

base_url = "http://127.0.0.1:8000/api/v1"

def test_query():
    print("Testing /query endpoint...")
    payload = {
        "question": "Explain RAG in simple terms.",
        "session_id": "verify_session_01"
    }
    response = requests.post(f"{base_url}/query", json=payload)
    if response.status_code == 200:
        print("Success!")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Failed: {response.status_code}")
        print(response.text)

def test_stream():
    print("\nTesting /chat/stream endpoint...")
    payload = {
        "question": "How does semantic chunking work?",
        "session_id": "verify_session_01"
    }
    response = requests.post(f"{base_url}/chat/stream", json=payload, stream=True)
    if response.status_code == 200:
        print("Success! Streaming response:")
        for line in response.iter_lines():
            if line:
                print(line.decode("utf-8"), end="", flush=True)
        print("\n")
    else:
        print(f"Failed: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    try:
        test_query()
        test_stream()
    except Exception as e:
        print(f"Error during testing: {e}")
