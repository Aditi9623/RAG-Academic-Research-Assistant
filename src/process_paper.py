import os
import pandas as pd

from pdf_processor import extract_text_from_pdf
from text_cleaner import clean_text
from chunker import create_chunks


# ==================================================
# PATHS
# ==================================================

PAPERS_FOLDER = "dataset/papers"

OUTPUT_PATH = (
    "dataset/processed/chunks.csv"
)

METADATA_PATH = (
    "dataset/processed/papers_metadata.csv"
)


# ==================================================
# CHUNK SETTINGS
# ==================================================

CHUNK_SIZE = 500
OVERLAP = 100


# ==================================================
# PROCESS ALL PAPERS
# ==================================================

def process_all_papers():

    all_chunks = []


    # ------------------------------------------------
    # Load paper metadata
    # ------------------------------------------------

    if os.path.exists(METADATA_PATH):

        metadata_df = pd.read_csv(
            METADATA_PATH
        )

        metadata_lookup = {}

        for _, row in metadata_df.iterrows():

            paper_number = int(
                row["paper_number"]
            )

            metadata_lookup[
                paper_number
            ] = row.to_dict()

    else:

        metadata_lookup = {}

        print(
            "Warning: papers_metadata.csv "
            "not found."
        )


    # ------------------------------------------------
    # Find PDFs
    # ------------------------------------------------

    pdf_files = [
        file_name
        for file_name in os.listdir(
            PAPERS_FOLDER
        )
        if file_name.lower().endswith(
            ".pdf"
        )
    ]

    pdf_files.sort()


    if not pdf_files:

        print(
            "No PDF files found in:",
            PAPERS_FOLDER
        )

        return


    print(
        f"Found {len(pdf_files)} PDF files."
    )


    # ==================================================
    # PROCESS EACH PDF
    # ==================================================

    for file_name in pdf_files:

        pdf_path = os.path.join(
            PAPERS_FOLDER,
            file_name
        )


        # ------------------------------------------------
        # Determine paper number
        # ------------------------------------------------

        try:

            paper_number = int(
                file_name
                .replace(
                    "paper_",
                    ""
                )
                .replace(
                    ".pdf",
                    ""
                )
            )

        except ValueError:

            print(
                f"Skipping invalid filename: "
                f"{file_name}"
            )

            continue


        paper_id = (
            f"P{paper_number:03d}"
        )


        # ------------------------------------------------
        # Metadata
        # ------------------------------------------------

        metadata = metadata_lookup.get(
            paper_number,
            {}
        )


        title = metadata.get(
            "title",
            ""
        )

        authors = metadata.get(
            "authors",
            ""
        )

        year = metadata.get(
            "year",
            ""
        )

        doi = metadata.get(
            "doi",
            ""
        )

        paper_url = metadata.get(
            "paper_url",
            ""
        )


        # ==================================================
        # DISPLAY PAPER
        # ==================================================

        print(
            "\n======================================"
        )

        print(
            f"Processing: {file_name}"
        )

        print(
            f"Paper ID: {paper_id}"
        )

        print(
            f"Title: {title}"
        )

        print(
            f"Year: {year}"
        )

        print(
            "======================================"
        )


        # ------------------------------------------------
        # Extract PDF text
        # ------------------------------------------------

        pages = extract_text_from_pdf(
            pdf_path
        )

        print(
            "Total pages:",
            len(pages)
        )


        paper_chunk_count = 0


        # ==================================================
        # PROCESS EACH PAGE
        # ==================================================

        for page in pages:

            page_number = page[
                "page_number"
            ]

            raw_text = page[
                "text"
            ]


            # ------------------------------------------------
            # Clean text
            # ------------------------------------------------

            cleaned_text = clean_text(
                raw_text
            )


            if not cleaned_text:

                continue


            # ------------------------------------------------
            # Create chunks
            # ------------------------------------------------

            chunks = create_chunks(
                cleaned_text,
                chunk_size=CHUNK_SIZE,
                overlap=OVERLAP
            )


            # ==================================================
            # STORE CHUNKS
            # ==================================================

            for chunk_number, chunk in enumerate(
                chunks,
                start=1
            ):

                chunk_record = {

                    "paper_id":
                        paper_id,

                    "file_name":
                        file_name,

                    "title":
                        title,

                    "authors":
                        authors,

                    "year":
                        year,

                    "doi":
                        doi,

                    "paper_url":
                        paper_url,

                    "page_number":
                        page_number,

                    "chunk_id":
                        (
                            f"{paper_id}_"
                            f"P{page_number}_"
                            f"C{chunk_number}"
                        ),

                    "text":
                        chunk
                }


                all_chunks.append(
                    chunk_record
                )


                paper_chunk_count += 1


        print(
            "Chunks created:",
            paper_chunk_count
        )


    # ==================================================
    # SAVE CHUNKS
    # ==================================================

    df = pd.DataFrame(
        all_chunks
    )


    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8"
    )


    # ==================================================
    # FINAL SUMMARY
    # ==================================================

    print(
        "\n======================================"
    )

    print(
        "PROCESSING COMPLETE"
    )

    print(
        "======================================"
    )

    print(
        "Total PDF papers:",
        len(pdf_files)
    )

    print(
        "Total chunks:",
        len(df)
    )

    print(
        "Processed dataset saved to:"
    )

    print(
        OUTPUT_PATH
    )


    # ==================================================
    # SHOW FIRST 3 CHUNKS
    # ==================================================

    print(
        "\n--- First 3 Chunks ---"
    )


    for _, row in df.head(3).iterrows():

        print(
            "\nChunk ID:",
            row["chunk_id"]
        )

        print(
            "Paper ID:",
            row["paper_id"]
        )

        print(
            "Title:",
            row["title"]
        )

        print(
            "Year:",
            row["year"]
        )

        print(
            "Page:",
            row["page_number"]
        )

        print(
            "Text:"
        )

        print(
            row["text"][:500]
        )


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    process_all_papers()