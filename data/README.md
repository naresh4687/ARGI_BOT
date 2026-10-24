# Knowledge-base data

```
data/
├── sources/manifest.json   # curated list of verified source URLs + metadata
├── raw/                    # source documents that get indexed (searched recursively)
│   ├── <id>.txt            # cleaned text of an HTML page
│   ├── <id>.pdf            # original PDF
│   └── <file>.meta.json    # sidecar: verified metadata for <file>
└── processed/              # built FAISS index (generated – do not edit)
```

Only files inside `data/raw/` are indexed. **Do not put notes or README files in
`data/raw/`** – every `.txt`, `.md`, `.json` and `.pdf` there becomes searchable
"evidence".

## Re-downloading the starter corpus

```bash
python -m knowledge_base.fetch_sources --allow-insecure-tls   # fetch the manifest URLs
python -m knowledge_base.ingest                                # rebuild the index
```

`--allow-insecure-tls` is needed because the TNAU server intermittently presents an
expired TLS certificate. Each affected file is marked `"tls_verified": false` in its
sidecar so you can see which downloads were not certificate-verified.

## Adding your own verified documents (works offline)

1. Download an official PDF (e.g. a TNAU crop production guide) and copy it to
   `data/raw/`, e.g. `data/raw/tnau-groundnut-guide.pdf`.
2. Create a sidecar file next to it named `tnau-groundnut-guide.pdf.meta.json`:

   ```json
   {
     "document_id": "tnau-groundnut-guide",
     "source_title": "Exact title as printed on the document",
     "source_url": "https://exact-url-you-downloaded-from-or-null",
     "publisher": "Tamil Nadu Agricultural University",
     "language": "en",
     "published_date": "2023-06 or null if not stated",
     "topic": "crop_management"
   }
   ```

   Only write values you can verify from the document or the page you downloaded it
   from. Use `null` for anything you do not know – **never guess a URL, title or
   date**. Without a sidecar the file name is used as the title and no URL is shown.
3. Run `python -m knowledge_base.ingest`. Files that are corrupt, encrypted, empty
   or scanned images (no extractable text) are skipped and listed in the output.

Supported formats: `.pdf` (text-based; page numbers are kept for citations),
`.txt`, `.md`, and `.json` in one of these shapes:

```json
{"source_title": "...", "language": "ta", "passages": [{"text": "...", "page_number": 4}]}
```

To add a FAQ, put a JSON file like the above in `data/raw/faq/`, with answers
taken from a verified source and that source recorded in the metadata.

## Licensing

See `data/sources/manifest.json` → `licensing_note`. The TNAU pages are marked
"© All Rights Reserved" and the portal gives no explicit reuse licence, so the
corpus is used as attributed excerpts for non-commercial education. Ask TNAU for
permission before redistributing it publicly (for example in a public GitHub
repository).
