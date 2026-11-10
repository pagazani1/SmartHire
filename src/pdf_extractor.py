"""Extract text from text-based PDF documents.

OCR is intentionally not performed. PDFs without an extractable text layer
raise :class:`NoExtractableTextError`.
"""

import re
from pathlib import Path

import pymupdf


class PDFExtractionError(Exception):
    """Base exception for PDF extraction failures."""


class InvalidPDFError(PDFExtractionError):
    """Raised when a path does not identify a readable PDF document."""


class EmptyPDFError(PDFExtractionError):
    """Raised when the PDF is empty or contains no pages."""


class PasswordProtectedPDFError(PDFExtractionError):
    """Raised when a PDF requires a password to read."""


class NoExtractableTextError(PDFExtractionError):
    """Raised when a PDF has no text layer; OCR is not implemented."""


def _clean_extracted_text(text: str) -> str:
    """Normalize extracted whitespace while retaining line and page order."""
    text = text.replace("\x00", "")
    lines = [line.strip() for line in text.splitlines()]
    cleaned_lines = []
    previous_line_was_blank = False

    for line in lines:
        if line:
            cleaned_lines.append(re.sub(r"[ \t]+", " ", line))
            previous_line_was_blank = False
        elif cleaned_lines and not previous_line_was_blank:
            cleaned_lines.append("")
            previous_line_was_blank = True

    return "\n".join(cleaned_lines).strip()


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Extract and clean text from a PDF, preserving page order.

    Args:
        pdf_path: Path to a PDF file.

    Returns:
        Extracted text, with pages separated by a newline.

    Raises:
        InvalidPDFError: The path is invalid, is not a PDF, or cannot be read.
        FileNotFoundError: The specified file does not exist.
        EmptyPDFError: The file is empty or has no pages.
        PasswordProtectedPDFError: The PDF requires a password.
        NoExtractableTextError: No text can be extracted; OCR is not implemented.
    """
    try:
        path = Path(pdf_path)
    except (TypeError, ValueError, OSError) as exc:
        raise InvalidPDFError("PDF path must be a valid filesystem path.") from exc

    try:
        if not path.exists():
            raise FileNotFoundError(f"PDF file does not exist: {path}")
        if not path.is_file():
            raise InvalidPDFError(f"PDF path is not a file: {path}")
        if path.suffix.lower() != ".pdf":
            raise InvalidPDFError("Input file must have a .pdf extension.")
        if path.stat().st_size == 0:
            raise EmptyPDFError(f"PDF file is empty: {path.name}")
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise InvalidPDFError(f"Unable to access PDF file: {path}") from exc

    try:
        document = pymupdf.open(path)
    except (RuntimeError, OSError, ValueError) as exc:
        if isinstance(exc, RuntimeError):
            raise InvalidPDFError(
                f"File is not a valid or readable PDF: {path.name}"
            ) from exc
        raise InvalidPDFError(f"Unable to read PDF file: {path.name}") from exc

    try:
        if document.is_encrypted or document.needs_pass:
            raise PasswordProtectedPDFError(
                f"PDF is password-protected and cannot be read: {path.name}"
            )
        if document.page_count == 0:
            raise EmptyPDFError(f"PDF contains no pages: {path.name}")

        page_text = [page.get_text() for page in document]
    except PDFExtractionError:
        raise
    except (RuntimeError, OSError, ValueError) as exc:
        raise PDFExtractionError(
            f"Failed to extract text from PDF: {path.name}"
        ) from exc
    finally:
        document.close()

    extracted_text = _clean_extracted_text("\n".join(page_text))
    if not extracted_text:
        raise NoExtractableTextError(
            f"No extractable text found in PDF: {path.name}. "
            "The PDF may be scanned or image-based; OCR is not implemented."
        )

    return extracted_text
