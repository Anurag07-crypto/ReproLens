import re
from pathlib import Path

from pypdf import PdfReader


SECTION_TERMS = (
    "experiment",
    "experimental setup",
    "implementation details",
    "training details",
    "appendix",
    "reproducibility",
)
HEADING = re.compile(
    r"(?im)^[ \t]*(?:(?:\d+(?:\.\d+)*\.?)[ \t]+)?"
    r"([A-Z][^\r\n]{2,100})[ \t]*$"
)


def extract_experiment_text(pdf_path: Path) -> str:
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Paper PDF not found: {pdf_path}")
    try:
        text = "\n".join(page.extract_text() or "" for page in PdfReader(str(pdf_path)).pages)
    except Exception as error:
        raise ValueError(f"Could not read paper PDF '{pdf_path}': {error}") from error
    if not text.strip():
        raise ValueError(f"No extractable text found in paper PDF: {pdf_path}")

    result = select_experiment_sections(text)
    if not result:
        raise ValueError(
            "Could not identify an experiment or appendix section. "
            "Check that the PDF contains extractable headings, or provide a text-based PDF."
        )
    return result


def select_experiment_sections(text: str) -> str:
    headings = [
        match
        for match in HEADING.finditer(text)
        if not re.search(r"[.!?][ \t]*$", match.group(1))
    ]
    selected: list[str] = []
    for index, match in enumerate(headings):
        title = match.group(1).strip().lower()
        if any(term in title for term in SECTION_TERMS):
            marker = re.match(r"^\s*(\d+(?:\.\d+)*)", match.group(0))
            depth = marker.group(1).count(".") + 1 if marker else None
            end = len(text)
            for following in headings[index + 1:]:
                next_number = re.match(r"^\s*(\d+(?:\.\d+)*)", following.group(0))
                if next_number is None or depth is None or next_number.group(1).count(".") + 1 <= depth:
                    end = following.start()
                    break
            selected.append(text[match.end():end].strip())
    return "\n\n".join(section for section in selected if section)
