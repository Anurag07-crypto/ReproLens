import re
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from langchain_community.document_loaders import PyPDFLoader


SECTION_TERMS = (
    "experiment",
    "implementation details",
    "training details",
    "appendix",
    "reproducibility",
    "dataset",
)
NUMBERED_HEADING = re.compile(
    r"(?im)^[ \t]*(?P<number>(?:\d+(?:\.\d+)*|[A-Z]\.\d+(?:\.\d+)*))[.)]?[ \t]+"
    r"(?P<title>[A-Z][^\r\n]{1,100})[ \t]*$"
)
INLINE_SECTION_HEADING = re.compile(
    r"(?i)(?P<number>\d+(?:\.\d+)*\.?)[ \t]+"
    r"(?P<title>experiments?|experimental setup|implementation details|"
    r"training details|reproducibility|appendix|datasets?)\b"
)
PLAIN_SECTION_HEADING = re.compile(r"(?im)^[ \t]*(?P<title>[A-Z][^\r\n]{2,80})[ \t]*$")


def _is_heading_title(title: str, max_words: int = 12) -> bool:
    return len(title.split()) <= max_words and not re.search(r"[.!?][ \t]*$", title)


def _heading_depth(number: str) -> int:
    return number.rstrip(".").count(".") + 1


def _is_section_boundary(current_number: str | None, next_number: str) -> bool:
    if current_number is None:
        return _heading_depth(next_number) == 1

    current_parts = current_number.rstrip(".").split(".")
    next_parts = next_number.rstrip(".").split(".")
    return current_parts[0] != next_parts[0] or len(next_parts) <= len(current_parts)


def extract_experiment_text(pdf_path: Path) -> str:
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Paper PDF not found: {pdf_path}")
    try:
        documents = PyPDFLoader(str(pdf_path), extraction_mode="layout").load()
        text = "\n".join(document.page_content for document in documents)
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
    headings: list[tuple[int, int, str | None, str]] = []
    for match in NUMBERED_HEADING.finditer(text):
        title = match.group("title").strip()
        if _is_heading_title(title):
            headings.append((match.start(), match.end(), match.group("number").rstrip("."), title))

    for match in PLAIN_SECTION_HEADING.finditer(text):
        title = match.group("title").strip()
        if (
            _is_heading_title(title, max_words=8)
            and any(term in title.lower() for term in SECTION_TERMS)
        ):
            headings.append((match.start(), match.end(), None, title))

    for match in INLINE_SECTION_HEADING.finditer(text):
        line_start = text.rfind("\n", 0, match.start()) + 1
        line_end = text.find("\n", match.end())
        if line_end == -1:
            line_end = len(text)
        title = match.group("title").strip()
        headings.append((line_start, line_end, match.group("number").rstrip("."), title))

    headings = sorted(
        set(headings),
        key=lambda heading: (heading[0], heading[1], heading[2] or "", heading[3]),
    )
    ranges: list[tuple[int, int]] = []
    for index, (_, heading_end, number, title) in enumerate(headings):
        if not any(term in title.lower() for term in SECTION_TERMS):
            continue

        end = len(text)
        for next_start, _, next_number, _ in headings[index + 1:]:
            if next_number is not None and _is_section_boundary(number, next_number):
                end = next_start
                break
        if heading_end < end:
            ranges.append((heading_end, end))

    merged_ranges: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged_ranges and start <= merged_ranges[-1][1]:
            previous_start, previous_end = merged_ranges[-1]
            merged_ranges[-1] = (previous_start, max(previous_end, end))
        else:
            merged_ranges.append((start, end))
    return "\n\n".join(
        section
        for start, end in merged_ranges
        if (section := text[start:end].strip())
    )
