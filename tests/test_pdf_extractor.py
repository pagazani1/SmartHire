import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pymupdf

from src.pdf_extractor import (
    EmptyPDFError,
    InvalidPDFError,
    NoExtractableTextError,
    PasswordProtectedPDFError,
    extract_text_from_pdf,
)


class TestExtractTextFromPDF(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_pdf(self, path: Path, page_texts: list[str]) -> None:
        document = pymupdf.open()
        for text in page_texts:
            page = document.new_page()
            page.insert_text((72, 72), text)
        document.save(path)
        document.close()

    def test_extracts_multiple_pages_in_order_and_cleans_whitespace(self):
        pdf_path = self.directory / "multi_page.pdf"
        self._create_pdf(
            pdf_path,
            ["First synthetic page", "Second synthetic page"],
        )

        extracted = extract_text_from_pdf(pdf_path)

        self.assertIn("First synthetic page", extracted)
        self.assertIn("Second synthetic page", extracted)
        self.assertLess(
            extracted.index("First synthetic page"),
            extracted.index("Second synthetic page"),
        )
        self.assertEqual(extracted, extracted.strip())

    def test_empty_file_raises_empty_pdf_error(self):
        pdf_path = self.directory / "empty.pdf"
        pdf_path.touch()

        with self.assertRaisesRegex(EmptyPDFError, "empty"):
            extract_text_from_pdf(pdf_path)

    def test_pdf_with_zero_pages_raises_empty_pdf_error(self):
        pdf_path = self.directory / "zero_pages.pdf"
        pdf_path.write_bytes(b"synthetic pdf placeholder")
        document = Mock()
        document.is_encrypted = False
        document.needs_pass = False
        document.page_count = 0

        with patch("src.pdf_extractor.pymupdf.open", return_value=document):
            with self.assertRaisesRegex(EmptyPDFError, "no pages"):
                extract_text_from_pdf(pdf_path)
        document.close.assert_called_once_with()

    def test_missing_path_raises_file_not_found_error(self):
        with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
            extract_text_from_pdf(self.directory / "missing.pdf")

    def test_invalid_path_type_raises_invalid_pdf_error(self):
        with self.assertRaisesRegex(InvalidPDFError, "valid filesystem path"):
            extract_text_from_pdf(None)

    def test_directory_path_raises_invalid_pdf_error(self):
        with self.assertRaisesRegex(InvalidPDFError, "not a file"):
            extract_text_from_pdf(self.directory)

    def test_non_pdf_extension_raises_invalid_pdf_error(self):
        text_path = self.directory / "not_a_pdf.txt"
        text_path.write_text("Synthetic non-PDF content", encoding="utf-8")

        with self.assertRaisesRegex(InvalidPDFError, r"\.pdf extension"):
            extract_text_from_pdf(text_path)

    def test_corrupted_pdf_raises_invalid_pdf_error(self):
        pdf_path = self.directory / "corrupted.pdf"
        pdf_path.write_bytes(b"not a valid PDF document")

        with self.assertRaisesRegex(InvalidPDFError, "not a valid or readable PDF"):
            extract_text_from_pdf(pdf_path)

    def test_password_protected_pdf_raises_clear_error(self):
        pdf_path = self.directory / "protected.pdf"
        document = pymupdf.open()
        page = document.new_page()
        page.insert_text((72, 72), "Synthetic protected document")
        document.save(
            pdf_path,
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            owner_pw="synthetic-owner",
            user_pw="synthetic-user",
        )
        document.close()

        with self.assertRaisesRegex(PasswordProtectedPDFError, "password-protected"):
            extract_text_from_pdf(pdf_path)

    def test_pdf_without_extractable_text_reports_ocr_not_implemented(self):
        pdf_path = self.directory / "no_text.pdf"
        document = pymupdf.open()
        document.new_page()
        document.save(pdf_path)
        document.close()

        with self.assertRaisesRegex(
            NoExtractableTextError, "OCR is not implemented"
        ):
            extract_text_from_pdf(pdf_path)


if __name__ == "__main__":
    unittest.main()
