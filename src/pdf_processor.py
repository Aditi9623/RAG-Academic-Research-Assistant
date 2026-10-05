import fitz
def extract_text_from_pdf(pdf_path):
    """
    Extract text from every page of an academic research paper.
    Page numbers are preserved for future citation support.
    """

    document = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text")

        pages.append({
            "page_number": page_number,
            "text": text
        })

    document.close()

    return pages