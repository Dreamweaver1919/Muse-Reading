## Upload Ingestion

Muse Reading now supports local ingestion for `txt`, `pdf`, and `epub` uploads without using external services.

### Supported formats

- `txt`
  - Decoded as UTF-8 text
- `pdf`
  - Parsed locally with `pypdf`
  - Extracted page text is normalized and forwarded into the existing `build_book_record` pipeline
- `epub`
  - Parsed locally with a zip/XML reader
  - The spine order is respected so chapter XHTML files are read in reading order

### Ingestion flow

1. `POST /api/upload` receives the file
2. `services/ingest/parser.py` detects the suffix and extracts readable text locally
3. Extracted text is normalized
4. The existing `build_book_record(...)` flow builds chapters and chunks
5. The existing temporal graph builder runs on the resulting `BookRecord`

### Current limits

- `pdf` extraction quality depends on whether the PDF contains selectable text
- Scanned PDFs without an OCR text layer are not yet supported
- `epub` support currently focuses on standard spine-based XHTML content
