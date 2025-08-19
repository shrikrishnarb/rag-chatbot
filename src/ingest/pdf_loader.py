import fitz  # PyMuPDF
import argparse
import json
from pathlib import Path


def extract_text_from_pdf(pdf_path: str):
    """
    Extract text and metadata (page numbers) from a PDF.
    Returns a list of dicts: [{"page": int, "text": str}, ...]
    """
    doc = fitz.open(pdf_path)
    pages = []

    for i, page in enumerate(doc, start=1):
        text = page.get_text("text")  # extract text in reading order
        pages.append({"page": i, "text": text.strip()})

    doc.close()
    return pages


def save_as_json(pages, out_path: str):
    """Save extracted pages as JSON file."""
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)


def save_as_txt(pages, out_path: str):
    """Save extracted pages as a plain text file (concatenated)."""
    with open(out_path, "w", encoding="utf-8") as f:
        for p in pages:
            f.write(f"--- Page {p['page']} ---\n")
            f.write(p["text"] + "\n\n")


def main():
    parser = argparse.ArgumentParser(description="Extract text from a PDF")
    parser.add_argument("pdf_path", help="Path to input PDF")
    parser.add_argument("--out", help="Output file (txt or json)", required=True)

    args = parser.parse_args()

    pages = extract_text_from_pdf(args.pdf_path)

    out_path = Path(args.out)
    if out_path.suffix == ".json":
        save_as_json(pages, args.out)
    elif out_path.suffix == ".txt":
        save_as_txt(pages, args.out)
    else:
        raise ValueError("Output file must end with .txt or .json")

    print(f"✅ Extracted text saved to {args.out}")


if __name__ == "__main__":
    main()
