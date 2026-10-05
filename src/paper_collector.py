import os
import requests
import pandas as pd


# ==================================================
# PATHS
# ==================================================

PAPERS_FOLDER = "dataset/papers"

METADATA_PATH = (
    "dataset/processed/papers_metadata.csv"
)

os.makedirs(
    PAPERS_FOLDER,
    exist_ok=True
)

os.makedirs(
    "dataset/processed",
    exist_ok=True
)


# ==================================================
# DOWNLOAD PDF
# ==================================================

def download_paper_pdf(
    pdf_url,
    paper_number
):

    if not pdf_url:
        return None

    try:

        response = requests.get(
            pdf_url,
            timeout=30,
            headers={
                "User-Agent":
                    "Mozilla/5.0 "
                    "(Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/154.0 Safari/537.36"
            }
        )

        response.raise_for_status()

        content_type = (
            response.headers
            .get(
                "Content-Type",
                ""
            )
            .lower()
        )

        is_pdf = (
            "application/pdf" in content_type
            or response.content[:4] == b"%PDF"
        )

        if not is_pdf:

            print(
                f"Skipped paper {paper_number}: "
                "response is not a PDF."
            )

            return None

        file_name = (
            f"paper_{paper_number:03d}.pdf"
        )

        file_path = os.path.join(
            PAPERS_FOLDER,
            file_name
        )

        with open(
            file_path,
            "wb"
        ) as file:

            file.write(
                response.content
            )

        print(
            f"Downloaded: {file_name}"
        )

        return file_path

    except requests.RequestException as error:

        print(
            f"Download failed for paper "
            f"{paper_number}: {error}"
        )

        return None


# ==================================================
# LOAD RANKED PAPERS
# ==================================================

def load_ranked_papers():

    if not os.path.exists(
        METADATA_PATH
    ):

        print(
            "\nERROR: Metadata file not found:"
        )

        print(
            METADATA_PATH
        )

        print(
            "\nPlease run:"
        )

        print(
            "python src\\academic_search.py"
        )

        return None

    try:

        df = pd.read_csv(
            METADATA_PATH
        )

    except Exception as error:

        print(
            "\nERROR: Could not read metadata:"
        )

        print(error)

        return None

    if df.empty:

        print(
            "\nERROR: Metadata file is empty."
        )

        return None

    return df


# ==================================================
# COLLECT PAPERS
# ==================================================

def collect_papers():

    print(
        "\nLoading ranked academic papers..."
    )

    metadata_df = load_ranked_papers()

    if metadata_df is None:
        return

    print(
        f"\nFound {len(metadata_df)} "
        "ranked papers in metadata."
    )

    print(
        "\nUsing papers from:"
    )

    print(
        METADATA_PATH
    )

    # --------------------------------------------------
    # Create string columns BEFORE assigning values.
    # This prevents pandas dtype errors when values
    # such as "" are assigned to originally empty columns.
    # --------------------------------------------------

    if "pdf_file" not in metadata_df.columns:

        metadata_df["pdf_file"] = ""

    else:

        metadata_df["pdf_file"] = (
            metadata_df["pdf_file"]
            .fillna("")
            .astype(str)
        )

    if "download_status" not in metadata_df.columns:

        metadata_df["download_status"] = ""

    else:

        metadata_df["download_status"] = (
            metadata_df["download_status"]
            .fillna("")
            .astype(str)
        )

    # Make sure paper numbers exist
    metadata_df["paper_number"] = range(
        1,
        len(metadata_df) + 1
    )

    print(
        "\nTrying papers with available PDFs..."
    )

    downloaded_count = 0

    # ==================================================
    # PROCESS EACH PAPER
    # ==================================================

    for index, row in metadata_df.iterrows():

        paper_number = index + 1

        title = str(
            row.get(
                "title",
                ""
            )
        )

        pdf_url = row.get(
            "pdf_url"
        )

        # --------------------------------------------------
        # Handle NaN / None safely
        # --------------------------------------------------

        if pd.isna(pdf_url):

            pdf_url = None

        else:

            pdf_url = str(
                pdf_url
            ).strip()

            if not pdf_url:
                pdf_url = None

        print(
            f"\nTrying paper {paper_number}:"
        )

        print(
            title
        )

        # --------------------------------------------------
        # No PDF URL
        # --------------------------------------------------

        if not pdf_url:

            print(
                f"No PDF URL available "
                f"for paper {paper_number}."
            )

            metadata_df.at[
                index,
                "pdf_file"
            ] = ""

            metadata_df.at[
                index,
                "download_status"
            ] = "No PDF URL"

            continue

        # --------------------------------------------------
        # Download
        # --------------------------------------------------

        downloaded_path = (
            download_paper_pdf(
                pdf_url,
                paper_number
            )
        )

        # --------------------------------------------------
        # Save status
        # --------------------------------------------------

        if downloaded_path:

            downloaded_count += 1

            pdf_file = (
                os.path.basename(
                    downloaded_path
                )
            )

            metadata_df.at[
                index,
                "pdf_file"
            ] = pdf_file

            metadata_df.at[
                index,
                "download_status"
            ] = "Downloaded"

        else:

            metadata_df.at[
                index,
                "pdf_file"
            ] = ""

            metadata_df.at[
                index,
                "download_status"
            ] = "Not Downloaded"

    # ==================================================
    # SAVE UPDATED METADATA
    # ==================================================

    metadata_df.to_csv(
        METADATA_PATH,
        index=False,
        encoding="utf-8"
    )

    # ==================================================
    # SUMMARY
    # ==================================================

    print(
        "\n======================================"
    )

    print(
        "PAPER COLLECTION SUMMARY"
    )

    print(
        "======================================"
    )

    print(
        f"Total ranked papers: "
        f"{len(metadata_df)}"
    )

    print(
        f"PDFs downloaded: "
        f"{downloaded_count}"
    )

    print(
        "\nMetadata updated:"
    )

    print(
        METADATA_PATH
    )

    # ==================================================
    # DOWNLOADED PAPERS
    # ==================================================

    downloaded_papers = metadata_df[
        metadata_df["download_status"]
        == "Downloaded"
    ]

    if len(downloaded_papers) > 0:

        print(
            "\nDownloaded papers:"
        )

        for _, row in (
            downloaded_papers.iterrows()
        ):

            print(
                f"\nPaper "
                f"{row['paper_number']}"
            )

            print(
                f"Title: "
                f"{row['title']}"
            )

            print(
                f"Year: "
                f"{row['year']}"
            )

            print(
                f"Relevance: "
                f"{row.get('relevance_category', 'N/A')}"
            )

            print(
                f"Relevance Score: "
                f"{row.get('relevance_score', 'N/A')}"
            )

            print(
                f"DOI: "
                f"{row['doi']}"
            )

            print(
                f"PDF: "
                f"{row['pdf_file']}"
            )

    else:

        print(
            "\nNo accessible PDFs were downloaded."
        )


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    print(
        "\n======================================"
    )

    print(
        "ACADEMIC PAPER COLLECTION"
    )

    print(
        "======================================"
    )

    print(
        "\nReading ranked papers from "
        "papers_metadata.csv..."
    )

    collect_papers()