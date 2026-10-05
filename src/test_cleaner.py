from text_cleaner import clean_text


sample_text = """
Retrieval-Augmented Generation

is a technique that combines
information retrieval with
large language models.
"""


cleaned_text = clean_text(sample_text)

print("Original Text:")
print(sample_text)

print("\nCleaned Text:")
print(cleaned_text)