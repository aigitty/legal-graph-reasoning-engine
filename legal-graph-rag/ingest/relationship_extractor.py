import json
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

SECTIONS_INPUT = PROCESSED_DIR / "sections.jsonl"
RELATIONSHIPS_OUTPUT = PROCESSED_DIR / "relationships.jsonl"


# Finds references like:
# section 25B
# sections 25C to 25E
# sec. 4
SECTION_REF_PATTERN = re.compile(
    r"\b(?:section|sections|sec\.|secs\.)\s+"
    r"(\d+[A-Z]{0,3}(?:-[A-Z]{1,3})?)"
    r"(?:\s*(?:to|-|and)\s*(\d+[A-Z]{0,3}(?:-[A-Z]{1,3})?))?",
    re.IGNORECASE,
)

# Finds references like:
# sub-section (1) of section 7
SUBSECTION_OF_SECTION_PATTERN = re.compile(
    r"\bsub-section\s*\([^)]+\)\s+of\s+section\s+"
    r"(\d+[A-Z]{0,3}(?:-[A-Z]{1,3})?)",
    re.IGNORECASE,
)

# If these words appear immediately after "section X of the ...",
# the reference is usually to an external statute/body, not the current Act.
EXTERNAL_REFERENCE_AFTER_PATTERN = re.compile(
    r"\bof\s+(?:the\s+)?[A-Z][A-Za-z&.,'’() -]{0,120}?\b"
    r"(Act|Code|Rules|Rule|Regulation|Regulations|Order|Ordinance|"
    r"Corporation|Commission|Authority|Board|Tribunal|Court|Constitution)\b",
    re.IGNORECASE,
)

# Known external law/body names commonly seen in Indian labour/employment statutes.
EXTERNAL_ACT_PATTERN = re.compile(
    r"\b("
    r"Companies Act|Company Act|Factories Act|Factory Act|Employees.? State Insurance Act|"
    r"Employees.? Provident Fund|Payment of Wages Act|Minimum Wages Act|Payment of Gratuity Act|"
    r"Industrial Disputes Act|Trade Unions Act|Income[- ]?tax Act|Indian Penal Code|"
    r"Code of Criminal Procedure|Criminal Procedure Code|Constitution of India|"
    r"Contract Labour|Shops and Commercial Establishments Act|Maternity Benefit Act|"
    r"Apprentices Act|Bonus Act|Railways Act|Mines Act|Plantations Labour Act|"
    r"Consumer Protection Act|Sexual Harassment of Women at Workplace|"
    r"Life Insurance Corporation Act|Food Corporations Act|Warehousing Corporations Act|"
    r"State Bank of India Act|Banking Companies|Airports Authority of India Act|"
    r"Dock Workers|Coal Mines Provident Fund|Industrial Finance Corporation|"
    r"National Housing Bank|Regional Rural Banks Act|Small Industries Development Bank|"
    r"Working Journalists|Sales Promotion Employees|"
    r"Karnataka Gazette|Official Gazette"
    r")\b",
    re.IGNORECASE,
)

# Phrases that usually indicate a same-Act/internal reference.
INTERNAL_REFERENCE_HINT_PATTERN = re.compile(
    r"\b("
    r"this Act|this Code|this Chapter|this section|this sub-section|"
    r"the foregoing provisions|the provisions of this Act|"
    r"under this Act|under this Code|for the purposes of this Act|"
    r"for the purposes of this section|in accordance with this Act|"
    r"as provided in|referred to in|specified in|mentioned in|"
    r"subject to|notwithstanding anything contained in|save as otherwise provided in|"
    r"under section|under sections|in section|in sections|"
    r"sub-section"
    r")\b",
    re.IGNORECASE,
)


def normalize_section_number(section_number: str) -> str:
    return (
        section_number
        .replace("-", "")
        .replace(".", "")
        .strip()
        .upper()
    )


def load_sections(path: Path) -> list[dict]:
    sections = []

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                sections.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error

    return sections


def build_section_lookup(sections: list[dict]) -> dict[tuple[str, str], str]:
    lookup = {}

    for section in sections:
        act_id = section["act_id"]
        section_number = normalize_section_number(section["section_number"])
        section_id = section["section_id"]

        lookup[(act_id, section_number)] = section_id

    return lookup


def extract_context(text: str, start: int, end: int, window: int = 140) -> str:
    context_start = max(0, start - window)
    context_end = min(len(text), end + window)

    return re.sub(r"\s+", " ", text[context_start:context_end]).strip()


def extract_after_reference(text: str, end: int, window: int = 160) -> str:
    """
    Returns text immediately after the matched reference.
    Useful for detecting external citations like:
    'section 3 of the Life Insurance Corporation Act'
    """
    return re.sub(r"\s+", " ", text[end:min(len(text), end + window)]).strip()


def is_probably_external_reference(
    context: str,
    after_reference: str,
    current_act_name: str,
    current_short_name: str,
) -> bool:
    """
    Skip references that clearly point to another Act, Code, Rule, authority, etc.

    Examples to skip:
    - section 3 of the Life Insurance Corporation Act
    - section 5 of the Companies Act, 1956
    - section 7 of the Factories Act
    - section 10 of the Code of Criminal Procedure

    Important:
    If the nearby text explicitly mentions the current act itself, we do not skip.
    """
    combined_text = f"{context} {after_reference}"
    combined_lower = combined_text.lower()

    current_act_lower = current_act_name.lower()
    current_short_lower = current_short_name.lower()

    if current_act_lower in combined_lower or current_short_lower in combined_lower:
        return False

    if EXTERNAL_REFERENCE_AFTER_PATTERN.search(after_reference):
        return True

    if EXTERNAL_ACT_PATTERN.search(combined_text):
        return True

    return False


def has_internal_reference_hint(context: str) -> bool:
    return bool(INTERNAL_REFERENCE_HINT_PATTERN.search(context))


def make_relationship(
    source_section: dict,
    target_section_id: str,
    target_section_number: str,
    evidence_text: str,
    relationship_type: str = "CITES",
) -> dict:
    return {
        "source_section_id": source_section["section_id"],
        "source_act_id": source_section["act_id"],
        "relationship_type": relationship_type,
        "target_section_id": target_section_id,
        "target_section_number": target_section_number,
        "evidence_text": evidence_text.strip(),
    }


def collect_reference_candidates(section_text: str) -> list[tuple[str, int, int]]:
    candidates = []

    for match in SECTION_REF_PATTERN.finditer(section_text):
        candidates.append((match.group(1), match.start(), match.end()))

        # Handles "sections 25C to 25E" by capturing both endpoints.
        # We are not expanding ranges yet because legal ranges can contain
        # alphanumeric sections like 25F, 25FF, 25FFA, etc.
        if match.group(2):
            candidates.append((match.group(2), match.start(), match.end()))

    for match in SUBSECTION_OF_SECTION_PATTERN.finditer(section_text):
        candidates.append((match.group(1), match.start(), match.end()))

    return candidates


def extract_relationships_from_section(
    section: dict,
    section_lookup: dict[tuple[str, str], str],
) -> list[dict]:
    relationships = []

    act_id = section["act_id"]
    source_section_id = section["section_id"]
    source_section_number = normalize_section_number(section["section_number"])
    section_text = section.get("section_text", "") or ""

    candidates = collect_reference_candidates(section_text)

    for raw_target_number, start, end in candidates:
        target_section_number = normalize_section_number(raw_target_number)

        if target_section_number == source_section_number:
            continue

        target_section_id = section_lookup.get((act_id, target_section_number))

        # If the target section does not exist in the same Act, do not create an edge.
        if not target_section_id:
            continue

        if target_section_id == source_section_id:
            continue

        evidence_text = extract_context(section_text, start, end)
        after_reference = extract_after_reference(section_text, end)

        if is_probably_external_reference(
            context=evidence_text,
            after_reference=after_reference,
            current_act_name=section["act_name"],
            current_short_name=section["short_name"],
        ):
            continue

        if not has_internal_reference_hint(evidence_text):
            continue

        relationships.append(
            make_relationship(
                source_section=section,
                target_section_id=target_section_id,
                target_section_number=target_section_number,
                evidence_text=evidence_text,
            )
        )

    return relationships


def deduplicate_relationships(relationships: list[dict]) -> list[dict]:
    unique = {}

    for relationship in relationships:
        key = (
            relationship["source_section_id"],
            relationship["relationship_type"],
            relationship["target_section_id"],
        )

        if key not in unique:
            unique[key] = relationship

    return list(unique.values())


def write_relationships(relationships: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for relationship in relationships:
            file.write(json.dumps(relationship, ensure_ascii=False) + "\n")


def main() -> None:
    sections = load_sections(SECTIONS_INPUT)
    section_lookup = build_section_lookup(sections)

    all_relationships = []

    for section in sections:
        section_relationships = extract_relationships_from_section(
            section=section,
            section_lookup=section_lookup,
        )
        all_relationships.extend(section_relationships)

    unique_relationships = deduplicate_relationships(all_relationships)
    write_relationships(unique_relationships, RELATIONSHIPS_OUTPUT)

    print("=" * 80)
    print("Relationship extraction completed")
    print(f"Input sections:        {len(sections)}")
    print(f"Relationships found:   {len(unique_relationships)}")
    print(f"Output file:           {RELATIONSHIPS_OUTPUT}")
    print("=" * 80)


if __name__ == "__main__":
    main()
