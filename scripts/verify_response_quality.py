import json
import time

import requests

BASE_URL = "http://localhost:8000/api/v1"
HEALTH_URL = "http://localhost:8000/health"  # Health remains at root
SESSION_ID = f"test_session_{int(time.time())}"


def test_response_quality():
    print(f"--- Starting Quality Verification ---")

    # 1. Check Health
    try:
        resp = requests.get(HEALTH_URL)
        print(f"Health Check: {resp.status_code}")
    except Exception as e:
        print(f"CRITICAL: API is not reachable at {HEALTH_URL}")
        return

    filename = "MODULE5.pdf"
    print(f"\n[Step 0] Using already uploaded {filename}...")

    # 3. Ask a question mentioning the file to establish context
    print(f"\n[Step 1] Querying with explicit filename {filename}...")
    payload = {
        "session_id": SESSION_ID,
        "question": f"What is in the file {filename}?",
    }
    resp = requests.post(f"{BASE_URL}/query", json=payload)
    result = resp.json()
    print(f"Response: {result['answer'][:100]}...")

    # 4. Ask for "bigger description" WITHOUT filename
    print("\n[Step 2] Asking for 'bigger description' of it (no filename)...")
    payload = {
        "session_id": SESSION_ID,
        "question": "Now give me a much bigger description of that file please",
    }
    resp = requests.post(f"{BASE_URL}/query", json=payload)
    result = resp.json()
    answer = result.get("answer", "")

    print(f"Response received. Model: {result.get('model')}")
    print("-" * 20)
    print(answer)
    print("-" * 20)

    # Validation
    # In the new implementation, "bigger" requests trigger a RAG search (not metadata retrieval)
    # and use a specialized system prompt for analysis.

    checks = {
        "Double Newlines": "\n\n" in answer,
        "Bold headers": "###" in answer or "**" in answer,
        "Deep Analysis Triggered": result.get("model") != "metadata-retrieval",
        "Length Check": len(answer) > 500,
    }

    print(f"\nVerification Results:")
    for check, passed in checks.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{check}: {status}")


if __name__ == "__main__":
    test_response_quality()
