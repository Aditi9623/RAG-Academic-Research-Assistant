import re


def clean_text(text):
    """
    Clean extracted academic paper text.

    Operations:
    1. Remove extra spaces
    2. Remove unnecessary line breaks
    3. Preserve meaningful text
    """

    # Replace multiple spaces and line breaks with a single space
    text = re.sub(r"\s+", " ", text)

    # Remove leading and trailing spaces
    text = text.strip()

    return text