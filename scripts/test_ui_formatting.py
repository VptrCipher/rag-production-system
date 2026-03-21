import json
import re

import requests


def test_formatting():
    print("--- 🔍 Testing Structured Readability ---")
    url = "http://localhost:8000/api/v1/agent/query"
    query = "Compare the self-attention mechanism in Transformers with the hybrid retrieval strategy we implemented in this RAG system. How do they both handle 'relevance'?"

    payload = {"question": query, "top_k": 5}

    try:
        print(f"Sending Agentic Query: {query}")
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        answer = data.get("answer", "")

        print("\n--- [START OF RESPONSE] ---")
        print(answer)
        print("--- [END OF RESPONSE] ---\n")

        # Check for structure
        has_headers = "##" in answer
        has_bolding = "**" in answer
        has_double_newlines = "\n\n" in answer

        print(f"✅ Has Headers (##): {has_headers}")
        print(f"✅ Has Bolding (**): {has_bolding}")
        print(f"✅ Has Double Newlines (\\n\\n): {has_double_newlines}")

        if has_headers and has_double_newlines:
            print("\n🌟 SUCCESS: The output is structured and Markdown-compliant.")
        else:
            print("\n⚠️ WARNING: Structure might be lacking. Check double newlines.")

    except Exception as e:
        print(f"❌ Error during test: {str(e)}")


if __name__ == "__main__":
    test_formatting()
