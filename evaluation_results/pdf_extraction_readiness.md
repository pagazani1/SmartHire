# SmartHire PDF Resume Extraction Readiness

## Summary

SmartHire does not currently have PDF extraction capability in its dependency
manifest, installed workspace Python environment, or inspected Python source
files. A reusable extractor would fit naturally in `src/pdf_extractor.py`,
with focused synthetic-PDF tests under `tests/`. Scanned/image-only PDFs need
an OCR path; no OCR Python package was found in the selected workspace
interpreter.

This was a read-only inspection. No package was installed and no actual
candidate resume was opened or processed.

## Existing dependencies

`requirements.txt` lists:

- `pandas>=2.0.0`
- `numpy>=1.24.0`
- `scikit-learn>=1.2.0`
- `datasets>=2.12.0`
- `pyarrow>=12.0.0`
- `joblib>=1.3.0`
- `matplotlib>=3.7.0`
- `seaborn>=0.12.0`

It does not declare PyMuPDF, pdfplumber, pypdf, PyPDF2, or an OCR dependency.

## Available PDF-related functionality

- The selected workspace interpreter's installed top-level module list
  included none of `fitz`/PyMuPDF, `pdfplumber`, `pypdf`, `PyPDF2`,
  `pytesseract`, `easyocr`, or `cv2`.
- The scanned Python source files have no PDF parser, OCR integration, or
  resume document-extraction function.
- Existing file-handling found in the inspected project is focused on
  Parquet/model artifacts and report files. References to a certificate PDF
  occur in background-verification descriptions/tests; they do not parse
  documents.
- `src/preprocessing.py`'s `clean_text()` processes already-extracted string
  values. It is not a PDF reader.

The installed-module result is specific to the Pylance-selected workspace
interpreter. This inspection did not establish whether an operating-system
OCR executable such as Tesseract is installed.

## Recommended library

**Recommendation: PyMuPDF (`pymupdf`) for initial text-based resume
extraction**, subject to confirming its licensing terms are acceptable for
SmartHire's intended distribution/use. It offers direct page text extraction,
page-level iteration, and a Windows-friendly Python API. It is not currently
installed or declared in the project.

If avoiding PyMuPDF's AGPL/commercial licensing model is a requirement,
evaluate `pypdf` instead; it is a pure-Python option for text-based PDFs but
may preserve layout less effectively. `pdfplumber` is another option when
layout/table inspection is important, though typical resumes are primarily
prose. None of these readers alone guarantees extraction from scanned pages.

Do not install a library until dependency and license selection has been
approved and added to the dependency manifest.

## Proposed extraction workflow

1. Accept an uploaded file/byte stream through a dedicated service boundary;
   do not trust a filename extension or client-supplied MIME type alone.
2. Enforce file-size and page-count limits before parsing. Check the PDF
   signature and reject unsupported formats.
3. Parse from bytes or a controlled temporary location with the chosen
   library. Close the document promptly; avoid retaining the original upload
   unless there is an explicit, secured retention requirement.
4. Extract text page by page in page order, preserving reasonable line breaks.
   Return structured metadata such as page count and extraction status
   alongside text; do not log resume text.
5. Normalize whitespace and pass the extracted string to the existing
   inference validation/cleaning path. Keep extraction separate from model
   prediction so extraction does not alter model behavior.
6. Detect likely image-only/low-text documents. Return an explicit
   `ocr_required` or equivalent status unless OCR has been configured.
7. If OCR is supported later, rasterize only within configured limits, use
   approved OCR language data, and treat OCR output as fallible text. Keep
   OCR optional and separately testable.
8. Convert parser/OCR failures into clear, safe errors for the caller, while
   avoiding disclosure of document contents in error messages or logs.

## Error cases to handle

- Empty upload or zero-byte input.
- Non-PDF input, malformed/truncated PDF, or a file with a misleading
  extension/MIME type.
- Password-protected/encrypted PDF.
- PDFs with no extractable text, blank pages, or only image content.
- Parser exceptions, unsupported PDF features, damaged page objects, or
  resource exhaustion.
- Excessive file size, page count, image dimensions, or extraction time.
- Unicode, unusual fonts/encodings, multi-column layouts, headers/footers,
  ligatures, and reading-order issues.
- Partial extraction: some pages readable while others fail or yield no text.
- OCR unavailable, missing language data, OCR timeout, or low-confidence
  output.
- Temporary-file cleanup failures, if temporary storage is used.

## OCR limitations

Scanned PDFs contain page images rather than a text layer, so a text extractor
may return empty or near-empty text even though the page appears readable.
Those cases require OCR, usually through an OCR engine such as Tesseract or
another separately selected service/library.

No OCR Python module appeared in the workspace interpreter's installed-module
list, and no OCR integration exists in the inspected source. The presence of a
system OCR executable was not checked. Therefore scanned-resume support is
**not available or verified** in the current project.

OCR can misread names, email addresses, dates, punctuation, columns, and
technical terms. It may also produce incomplete or reordered text. A future
implementation should report OCR use/failure, avoid treating OCR output as
ground truth, and test with synthetic documents rather than real candidate
resumes during development.

## Tests needed

No PDF extraction tests currently exist in the inspected `tests/` directory.
Before integration, add focused tests using generated/synthetic PDFs:

1. Extract expected text from a small text-based PDF with multiple pages and
   verify page order and whitespace normalization.
2. Handle empty, corrupt, non-PDF, and encrypted inputs with documented
   errors.
3. Confirm image-only/scanned input is detected as requiring OCR, not silently
   returned as a successful empty resume.
4. Test size/page limits and ensure documents/temp files are closed and cleaned
   up on both success and failure.
5. Test Unicode, punctuation, columns/reading order, and partial-page
   extraction using synthetic content only.
6. If OCR is added, test missing executable/language data, timeout/failure,
   and expected OCR status without asserting perfect transcription.
7. Test the integration boundary from extracted text into existing input
   validation and prediction preprocessing, without changing the model's
   TF-IDF or classifier behavior.

These tests were not run because no extraction implementation or PDF library
is present, and installing packages was explicitly out of scope.
