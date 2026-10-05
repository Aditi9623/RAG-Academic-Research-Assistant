from pdf_processor import extract_text_from_pdf


pdf_path = "dataset/papers/paper_001.pdf"

pages = extract_text_from_pdf(pdf_path)

print("Total pages:", len(pages))

print("\n--- First Page Text ---")
print(pages[0]["text"][:2000])