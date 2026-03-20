import requests
import os
import time

base_url = "http://127.0.0.1:8000/api/v1"

def test_upload_and_summarize():
    print("--- Testing File Upload and Summarization ---")
    
    # Create a dummy PDF for testing if one doesn't exist
    test_pdf = "test_document.pdf"
    if not os.path.exists(test_pdf):
        with open(test_pdf, "wb") as f:
            f.write(b"%PDF-1.4\n1 0 obj\n<< /Title (Test) >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF")
    
    try:
        # 1. Test Upload
        print(f"Uploading {test_pdf}...")
        with open(test_pdf, "rb") as f:
            files = {"file": (test_pdf, f, "application/pdf")}
            r = requests.post(f"{base_url}/upload", files=files)
        
        if r.status_code == 200:
            data = r.json()
            print("Upload Success!")
            print(f"Short Description: {data.get('short_description')}")
        else:
            print(f"Upload Failed: {r.status_code}")
            print(r.text)
            return

        # 2. Test Short Description Retrieval
        print("\nQuerying for short description...")
        payload = {
            "question": f"Give me a short description of {test_pdf}",
            "session_id": "test_verification"
        }
        r = requests.post(f"{base_url}/query", json=payload)
        if r.status_code == 200:
            print(f"Bot response: {r.json().get('answer')}")
        else:
            print(f"Query failed: {r.status_code}")

        # 3. Test Long Description Retrieval
        print("\nQuerying for long description...")
        payload["question"] = f"Give me a long description of {test_pdf}"
        r = requests.post(f"{base_url}/query", json=payload)
        if r.status_code == 200:
            print(f"Bot response: {r.json().get('answer')}")
        else:
            print(f"Query failed: {r.status_code}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_upload_and_summarize()
