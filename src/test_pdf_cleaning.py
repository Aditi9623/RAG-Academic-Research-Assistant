from pdf_processor import extract_text_from_pdf
from text_cleaner import clean_text


pdf_path = "dataset/papers/paper_001.pdf"

pages = extract_text_from_pdf(pdf_path)

print("Total pages:", len(pages))

print("\n--- CLEANED TEXT FROM FIRST PAGE ---")

first_page_text = pages[0]["text"]

cleaned_text = clean_text(first_page_text)

print(cleaned_text[:3000])