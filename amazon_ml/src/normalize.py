import re
import unicodedata
from typing import List

try:
    from unidecode import unidecode
except ImportError:
    unidecode = None


# Common legal/business suffixes.
LEGAL_SUFFIXES = {
    "private limited",
    "pvt limited",
    "pvt ltd",
    "private ltd",
    "limited",
    "ltd",
    "corporation",
    "corp",
    "incorporated",
    "inc",
    "llc",
    "pllc",
    "l l c",
    "co",
    "company",
}


def normalize_basic(text: str) -> str:
    """
    Unicode-safe normalization.

    Preserves letters, combining marks, and numbers from
    non-Latin scripts while normalizing punctuation and spaces.
    """
    if text is None:
        return ""

    text = str(text).strip()

    if not text:
        return ""

    # NFC keeps characters + combining marks in a proper Unicode form.
    text = unicodedata.normalize("NFC", text).lower()

    # Convert ampersand to a useful word.
    text = text.replace("&", " and ")

    result = []

    for char in text:
        category = unicodedata.category(char)

        # Letters: L*
        # Marks:   M*
        # Numbers: N*
        # Whitespace: Z*
        if (
            category.startswith("L")
            or category.startswith("M")
            or category.startswith("N")
            or category.startswith("Z")
        ):
            result.append(char)
        else:
            # Punctuation/symbols become spaces.
            result.append(" ")

    text = "".join(result)

    # Collapse repeated whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_compact(text: str) -> str:
    """
    Compact representation useful for exact/prefix blocking.
    """
    text = normalize_basic(text)
    return re.sub(r"\s+", "", text)


def remove_legal_suffix(text: str) -> str:
    """
    Remove trailing legal/business suffixes while preserving
    the rest of the business name.
    """
    text = normalize_basic(text)

    if not text:
        return ""

    tokens = text.split()

    while tokens:
        removed = False

        for suffix in sorted(
            LEGAL_SUFFIXES,
            key=lambda x: len(x.split()),
            reverse=True,
        ):
            suffix_tokens = suffix.split()

            if len(tokens) >= len(suffix_tokens):
                if tokens[-len(suffix_tokens):] == suffix_tokens:
                    tokens = tokens[:-len(suffix_tokens)]
                    removed = True
                    break

        if not removed:
            break

    return " ".join(tokens)


def transliterate(text: str) -> str:
    """
    Convert non-Latin scripts to a Latin representation.

    The original Unicode value is always retained separately.
    """
    if text is None:
        return ""

    text = str(text).strip()

    if not text:
        return ""

    if unidecode is None:
        return normalize_basic(text)

    text = unidecode(text).lower()

    # Normalize whitespace without applying Unicode \w processing
    # to the transliterated output.
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()



def extract_numbers(text: str) -> List[str]:
    """
    Extract numeric sequences from addresses/names.

    Example:
        '12 MG Road, Bangalore 560037'
        -> ['12', '560037']
    """
    if not text:
        return []

    return re.findall(r"\d+", str(text))


def normalize_address(text: str) -> str:
    """
    Address-specific normalization.
    """
    return normalize_basic(text)


def make_name_features(text: str) -> dict:
    """
    Create multiple name representations.
    """
    basic = normalize_basic(text)

    return {
        "name_raw": "" if text is None else str(text),
        "name_basic": basic,
        "name_compact": normalize_compact(text),
        "name_no_suffix": remove_legal_suffix(text),
        "name_transliterated": transliterate(text),
    }


def make_address_features(text: str) -> dict:
    """
    Create multiple address representations.
    """
    basic = normalize_address(text)
    numbers = extract_numbers(text)

    return {
        "address_raw": "" if text is None else str(text),
        "address_basic": basic,
        "address_compact": normalize_compact(text),
        "address_numbers": " ".join(numbers),
        "address_number_tokens": numbers,
    }


if __name__ == "__main__":
    examples = [
        "Shree Ganesh Electronics Pvt. Ltd.",
        "ABC & Sons Corporation",
        "सुप्रीम आईटी प्राइवेट लिमिटेड",
    ]

    for x in examples:
        print("\nRAW :", x)
        print("BASIC:", normalize_basic(x))
        print("COMPACT:", normalize_compact(x))
        print("NO SUFFIX:", remove_legal_suffix(x))
        print("TRANSLIT:", transliterate(x))

    address = "12, M.G. Road, Bengaluru 560037"
    print("\nADDRESS:", address)
    print("NORMALIZED:", normalize_address(address))
    print("NUMBERS:", extract_numbers(address))