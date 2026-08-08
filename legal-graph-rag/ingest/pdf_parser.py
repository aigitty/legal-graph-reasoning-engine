import json
import re
from pathlib import Path

import pdfplumber


BASE_DIR = Path(__file__).resolve().parents[1]
ACTS_DIR = BASE_DIR / "data" / "acts"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
SECTIONS_OUTPUT = PROCESSED_DIR / "sections.jsonl"


PDF_CONFIG = {
    "industrial_disputes_act_1947.pdf": {
        "act_id": "IDA_1947",
        "act_name": "Industrial Disputes Act",
        "year": 1947,
        "jurisdiction": "Central",
        "short_name": "IDA",
        "parser_mode": "arrangement_based",
        "start_marker": "THE INDUSTRIAL DISPUTES ACT, 1947",
        "stop_markers": ["THE FIRST SCHEDULE"],
    },
    "code_on_wages_2019.pdf": {
        "act_id": "COW_2019",
        "act_name": "Code on Wages",
        "year": 2019,
        "jurisdiction": "Central",
        "short_name": "COW",
        "parser_mode": "arrangement_based",
        "start_marker": "THE CODE ON WAGES, 2019",
        "stop_markers": [],
    },
    # ------------------------------------------------------------------ #
    # The other two Labour Codes brought into force on 21 November 2025
    # (S.O. 5320(E)). Together with the Code on Wages 2019 above they
    # replace the Industrial Disputes Act 1947 (IRC s.104) and the Payment
    # of Gratuity Act 1972 (SSC s.164).
    #
    # STOP-MARKER NOTE — do not "simplify" these back to "THE FIRST SCHEDULE".
    # cut_at_stop_markers() matches a multi-word marker ANYWHERE in the body,
    # not just as a heading, and both Codes reference their own First Schedule
    # inline in early sections (e.g. IRC s.2(zj)). Using that marker truncated
    # the IRC body from 197,000 characters to 15,000 and yielded 2 sections out
    # of 104. The markers below were each verified to occur EXACTLY ONCE in the
    # body, at the real start of the schedules.
    # ------------------------------------------------------------------ #
    "The_Industrial_Relations_Code_2020.pdf": {
        "act_id": "IRC_2020",
        "act_name": "Industrial Relations Code",
        "year": 2020,
        "jurisdiction": "Central",
        "short_name": "IRC",
        "parser_mode": "arrangement_based",
        "start_marker": "THE INDUSTRIAL RELATIONS CODE, 2020",
        "stop_markers": ["MATTERS TO BE PROVIDED IN STANDING ORDERS"],
    },
    "The_Code_on_Social_Security_2020.pdf": {
        "act_id": "SSC_2020",
        "act_name": "Code on Social Security",
        "year": 2020,
        "jurisdiction": "Central",
        "short_name": "SSC",
        "parser_mode": "arrangement_based",
        "start_marker": "THE CODE ON SOCIAL SECURITY, 2020",
        "stop_markers": ["Chapter No. Chapter Heading Applicability"],
    },
    "karnataka_shops_act_1961.pdf": {
        "act_id": "KSEA_1961",
        "act_name": "Karnataka Shops and Commercial Establishments Act",
        "year": 1961,
        "jurisdiction": "Karnataka",
        "short_name": "KSEA",
        "parser_mode": "arrangement_based",
        "start_marker": "THE 1[KARNATAKA]1 SHOPS AND COMMERCIAL ESTABLISHMENTS",
        "fallback_start_marker": "CHAPTER I \nPRELIMINARY",
        "stop_markers": ["SCHEDULE"],
        "expected_sections": [
            "1", "2", "3", "4", "5", "6", "6A", "7", "8", "9", "10",
            "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
            "21", "22", "23", "24", "25", "26", "27", "28", "29", "30",
            "31", "32", "33", "33A", "34", "35", "36", "37", "38", "39",
            "40", "41", "42", "43", "44",
        ],
    },
    "minimum_wages_act_1948.pdf": {
        "act_id": "MWA_1948",
        "act_name": "Minimum Wages Act",
        "year": 1948,
        "jurisdiction": "Central",
        "short_name": "MWA",
        "parser_mode": "body_only",
        "start_marker": "THE MINIMUM WAGES ACT, 1948",
        "stop_markers": ["SCHEDULE", "THE MINIMUM WAGES (CENTRAL) RULES"],
        "expected_sections": [
            "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
            "11", "12", "13", "14", "15", "16", "17", "18", "19",
            "20", "21", "22", "22A", "22B", "22C", "22D", "22E",
            "22F", "23", "24", "25", "26", "27", "28", "29", "30",
            "30A", "31",
        ],
    },
    "payment_of_gratuity_act_1972.pdf": {
        "act_id": "PGA_1972",
        "act_name": "Payment of Gratuity Act",
        "year": 1972,
        "jurisdiction": "Central",
        "short_name": "PGA",
        "parser_mode": "body_only",
        "start_marker": "THE PAYMENT OF GRATUITY ACT, 1972",
        "stop_markers": ["Rule 5","THE PAYMENT OF GRATUITY (CENTRAL) RULES"],
        "expected_sections": [
            "1", "2", "2A", "3", "4", "4A", "5", "6", "7", "7A",
            "7B", "8", "9", "10", "11", "12", "13", "13A", "14", "15",
        ],
    },
}


SECTION_START_PATTERN = re.compile(
    r"^\s*\[?\s*(\d+[A-Z]{0,3}(?:-[A-Z]{1,3})?\s*\.|\d+(?:[A-Z]{1,3}|-[A-Z]{1,3}))\s*(.*)$"
)

SCHEDULE_HEADING_PATTERN = re.compile(
    r"^\s*THE\s+FIRST\s+SCHEDULE", re.IGNORECASE
)


def normalize_section_number(section_number: str) -> str:
    return (
        section_number
        .replace("-", "")
        .replace(".", "")
        .strip()
    )


def clean_text(text: str) -> str:
    text = text.replace("\x0c", " ")
    text = re.sub(r"-\s+", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^\s*\d+\s*\[\s*", "[", line)
    line = re.sub(r"\]+$", "", line)
    return line.strip()


def is_noise_line(line: str) -> bool:
    upper = line.strip().upper()

    if not line.strip():
        return True

    if re.fullmatch(r"\d+", line.strip()):
        return True

    if upper == "SECTIONS":
        return True

    if upper.startswith("CHAPTER "):
        return True

    if upper.startswith("STATEMENT OF OBJECTS"):
        return True

    return False


def is_footnote_line(line: str) -> bool:
    lower = line.lower()

    markers = [
        "subs. by",
        "substituted by",
        "ins. by",
        "inserted by",
        "omitted by",
        "rep. by",
        "w.e.f.",
        "vide ",
        "the words ",
        "the expression ",
        "shall stand substituted",
    ]

    return any(marker in lower for marker in markers)


def extract_pdf_text(pdf_path: Path) -> str:
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            pages.append(page_text)

    full_text = "\n".join(pages)
    full_text = re.sub(r"-\n", "", full_text)

    full_text = normalize_inline_section_starts(full_text)

    return full_text


def normalize_inline_section_starts(text: str) -> str:
    # Some PDFs extract "as follows:1. Short title..." on the same line.
    # Move the first section header to a new line so the section parser can detect it.
    text = re.sub(
        r"(?i)(as\s+follows\s*:?\s*-?)\s*(\d+[A-Z]{0,3}\.)",
        r"\1\n\2",
        text,
    )

    text = re.sub(
        r"(?i)(it\s+is\s+hereby\s+enacted\s+as\s+follows\s*:?\s*-?)\s*(\d+[A-Z]{0,3}\.)",
        r"\1\n\2",
        text,
    )

    return text


def cut_at_stop_markers(text: str, stop_markers: list[str]) -> str:
    if not stop_markers:
        return text

    upper_text = text.upper()

    for marker in stop_markers:
        marker_upper = marker.upper()

        # For short markers like SCHEDULE, only cut when it appears as a heading line.
        if len(marker_upper.split()) == 1:
            pattern = rf"(?m)^\s*{re.escape(marker_upper)}\s*$"
        else:
            # For long markers like THE PAYMENT OF GRATUITY (CENTRAL) RULES,
            # allow whitespace/newline breaks inside the heading.
            pattern = r"\s+".join(
                re.escape(part) for part in marker_upper.split()
            )

        match = re.search(pattern, upper_text, flags=re.IGNORECASE)

        if match:
            return text[:match.start()]

    return text

def split_arrangement_and_body(raw_text: str, metadata: dict):
    start_marker = metadata["start_marker"]
    positions = [m.start() for m in re.finditer(re.escape(start_marker), raw_text)]

    if len(positions) >= 2:
        arrangement_text = raw_text[:positions[1]]
        body_text = raw_text[positions[1]:]
        return arrangement_text, body_text

    fallback_marker = metadata.get("fallback_start_marker")

    if fallback_marker and fallback_marker in raw_text:
        body_start = raw_text.find(fallback_marker)
        return raw_text[:body_start], raw_text[body_start:]

    if positions:
        return "", raw_text[positions[0]:]

    return "", raw_text


def trim_body_only_text(raw_text: str, metadata: dict):
    start_marker = metadata["start_marker"]
    start_pos = raw_text.upper().find(start_marker.upper())

    if start_pos != -1:
        raw_text = raw_text[start_pos:]

    raw_text = cut_at_stop_markers(raw_text, metadata.get("stop_markers", []))

    return raw_text


def extract_expected_sections_from_arrangement(arrangement_text: str):
    expected_sections = []

    for line in arrangement_text.splitlines():
        line = clean_line(line.strip())

        if not line:
            continue

        if SCHEDULE_HEADING_PATTERN.match(line):
            break

        if is_noise_line(line):
            continue

        match = SECTION_START_PATTERN.match(line)

        if match:
            section_number = normalize_section_number(match.group(1))

            if section_number not in expected_sections:
                expected_sections.append(section_number)

    return expected_sections


def split_title_and_body(section_buffer: str):
    text = clean_text(section_buffer)

    split_patterns = [
        r"\.\s*—\s*",
        r"\.—\s*",
        r"\.\s*-\s*",
        r"\s+-\s+",
        r"\s+—\s+",
    ]

    for pattern in split_patterns:
        parts = re.split(pattern, text, maxsplit=1)

        if len(parts) == 2:
            return clean_text(parts[0]), clean_text(parts[1])

    sentence_split = re.split(r"\.\s+", text, maxsplit=1)

    if len(sentence_split) == 2:
        return clean_text(sentence_split[0]), clean_text(sentence_split[1])

    return "", text


def build_section_record(section_number, buffer_lines, metadata, source_file):
    full_buffer = " ".join(buffer_lines)
    section_title, section_text = split_title_and_body(full_buffer)
    clean_number = normalize_section_number(section_number)

    return {
        "act_id": metadata["act_id"],
        "act_name": metadata["act_name"],
        "year": metadata["year"],
        "jurisdiction": metadata["jurisdiction"],
        "short_name": metadata["short_name"],
        "section_id": f'{metadata["act_id"]}_S{clean_number}',
        "section_number": clean_number,
        "section_title": section_title,
        "section_text": section_text,
        "source_file": source_file,
    }


def parse_with_expected_sections(body_text: str, expected_sections: list[str], metadata: dict, source_file: str):
    expected_index = 0
    sections = []

    current_section_number = None
    current_buffer = []
    in_state_amendment = False

    for raw_line in body_text.splitlines():
        line = clean_line(raw_line.strip())

        if not line:
            continue

        match = SECTION_START_PATTERN.match(line)

        # Detect section headers before footnote filtering.
        if match:
            candidate_number = normalize_section_number(match.group(1))
            rest_of_line = match.group(2).strip()

            expected_next = (
                expected_sections[expected_index]
                if expected_index < len(expected_sections)
                else None
            )

            if candidate_number == expected_next:
                if current_section_number and current_buffer:
                    sections.append(
                        build_section_record(
                            current_section_number,
                            current_buffer,
                            metadata,
                            source_file,
                        )
                    )

                current_section_number = candidate_number
                current_buffer = [rest_of_line]
                expected_index += 1
                in_state_amendment = False
                continue

        upper = line.upper()

        if upper == "STATE AMENDMENT":
            in_state_amendment = True
            continue

        if in_state_amendment:
            continue

        if is_noise_line(line):
            continue

        if is_footnote_line(line):
            continue

        if current_section_number:
            current_buffer.append(line)

    if current_section_number and current_buffer:
        sections.append(
            build_section_record(
                current_section_number,
                current_buffer,
                metadata,
                source_file,
            )
        )

    found_sections = [s["section_number"] for s in sections]
    missing_sections = [s for s in expected_sections if s not in found_sections]

    if missing_sections:
        print(f"Missing sections:     {missing_sections}")
    else:
        print("Missing sections:     None")

    return sections


def parse_arrangement_based(raw_text: str, metadata: dict, source_file: str):
    arrangement_text, body_text = split_arrangement_and_body(raw_text, metadata)

    # Some PDFs have an Arrangement of Sections, but the extracted text is not
    # stable enough to rely on it. For those PDFs, use the manually supplied
    # section list but still parse only from the real Act body.
    expected_sections = metadata.get("expected_sections")

    if not expected_sections:
        expected_sections = extract_expected_sections_from_arrangement(arrangement_text)

    if not expected_sections:
        raise ValueError(
            f"Could not extract expected sections from Arrangement of Sections for {source_file}."
        )

    body_text = cut_at_stop_markers(
        body_text,
        metadata.get("stop_markers", []),
    )

    print(f"Expected sections:    {len(expected_sections)}")

    return parse_with_expected_sections(
        body_text,
        expected_sections,
        metadata,
        source_file,
    )


def parse_body_only(raw_text: str, metadata: dict, source_file: str):
    body_text = trim_body_only_text(raw_text, metadata)
    expected_sections = metadata["expected_sections"]

    print(f"Expected sections:    {len(expected_sections)}")

    return parse_with_expected_sections(
        body_text,
        expected_sections,
        metadata,
        source_file,
    )


def parse_sections(raw_text: str, metadata: dict, source_file: str):
    parser_mode = metadata["parser_mode"]

    if parser_mode == "arrangement_based":
        return parse_arrangement_based(raw_text, metadata, source_file)

    if parser_mode == "body_only":
        return parse_body_only(raw_text, metadata, source_file)

    raise ValueError(f"Unknown parser_mode: {parser_mode}")


def write_jsonl(records, output_path: Path):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    all_sections = []

    for pdf_name, metadata in PDF_CONFIG.items():
        pdf_path = ACTS_DIR / pdf_name

        if not pdf_path.exists():
            print(f"Skipping missing PDF: {pdf_path}")
            continue

        print("\n" + "=" * 80)
        print(f"Parsing:              {pdf_name}")

        raw_text = extract_pdf_text(pdf_path)
        sections = parse_sections(raw_text, metadata, pdf_name)

        print(f"Extracted sections:   {len(sections)}")

        all_sections.extend(sections)

    write_jsonl(all_sections, SECTIONS_OUTPUT)

    print("\n" + "=" * 80)
    print(f"Written to: {SECTIONS_OUTPUT}")
    print(f"Total sections:       {len(all_sections)}")


if __name__ == "__main__":
    main()