import re


def clean_markdown_simulation(text):
    if not text:
        return ""
    # 1. Fix missing newlines before headers
    text = re.sub(r"([^\n])(#{1,6}\s)", r"\1\n\n\2", text)

    # 2. Fix citation clusters
    text = re.sub(r"\[Source\s*(\d+)\s*,\s*Source\s*(\d+)\]", r"[Source \1] [Source \2]", text)
    text = re.sub(r"\[Source\s*,\s*Source\s*,\s*Source\s*\]", r"[Source 1] [Source 2] [Source 3]", text)
    text = re.sub(r"\[Source\s*,\s*Source\s*\]", r"[Source 1] [Source 2]", text)
    text = re.sub(r"\[Source\s*,\s*\]", r"[Source 1]", text)

    # 3. Fix mashed citations
    text = re.sub(r"([a-zA-Z0-9])(\[Source)", r"\1 \2", text)

    return text


# Test cases from user screenshot
test_text = "Input/Output Organization [Source ]. ## Key Concepts\nSome key concepts covered in Module include:. Program Controlled I/O: [Source ]. Interrupt Controlled I/O: [Source ]. ## Device Controllers\nThe device controller [Source ]. [Source , Source , Source ]."

cleaned = clean_markdown_simulation(test_text)
print("--- CLEANED OUTPUT ---")
print(cleaned)

# Verification assertions
assert "Organization [Source ]. \n\n## Key Concepts" in cleaned, "Header fix failed"
assert "[Source 1] [Source 2] [Source 3]" in cleaned, "Citation cluster fix failed"
print("\nVerification Passed!")
