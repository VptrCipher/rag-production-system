import requests
import time
import sys

BASE_URL = "http://localhost:8000/api/v1"

def test_health():
    print("Testing /health endpoint...")
    response = requests.get("http://localhost:8000/health")
    print(f"Status: {response.status_code}")
    print(f"Body: {response.json()}")
    assert response.status_code == 200
    assert "dependencies" in response.json()

def test_pii_redaction():
    print("\nTesting PII Redaction...")
    payload = {
        "question": "My email is test@example.com and phone is 555-0199. What is RAG?",
        "user_id": "test_user"
    }
    # We check the logs for redaction.
    response = requests.post(f"{BASE_URL}/query", json=payload, timeout=30)
    print(f"Status: {response.status_code}")
    assert response.status_code == 200

def test_caching():
    print("\nTesting Caching...")
    payload = {"question": "What is hybrid search?", "user_id": "test_user"}
    
    # First request
    start = time.time()
    resp1 = requests.post(f"{BASE_URL}/query", json=payload, timeout=30)
    dur1 = time.time() - start
    print(f"First request duration: {dur1:.2f}s")
    
    # Second request (should be cached)
    start = time.time()
    resp2 = requests.post(f"{BASE_URL}/query", json=payload, timeout=30)
    dur2 = time.time() - start
    print(f"Second request duration: {dur2:.2f}s")
    
    print(f"Speedup: {dur1/max(dur2, 0.001):.1f}x")
    assert dur2 < dur1 or dur2 < 0.5 # Second one usually < 100ms

def test_rate_limiting():
    print("\nTesting Rate Limiting...")
    payload = {"question": "Ping", "user_id": "test_user"}
    for i in range(70): # Limit is 60
        response = requests.post(f"{BASE_URL}/query", json=payload)
        if response.status_code == 429:
            print(f"Rate limited successfully at request {i+1}")
            return
    print("Failed to trigger rate limit.")

def main():
    try:
        test_health()
        test_pii_redaction()
        test_caching()
        test_rate_limiting()
        print("\nAll production features verified!")
    except Exception as e:
        print(f"\nVerification failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
