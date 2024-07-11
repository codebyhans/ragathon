# RAGathon

A Python project focused on building a Retrieval Augmented Generation (RAG) pipeline from scratch. It's designed for a two-day hackathon or workshop to provide hands-on experience with RAG implementation.

## Development Setup

1. Ensure [pyenv](https://github.com/pyenv/pyenv) and [PDM](https://pdm.fming.dev/) are installed.

2. Install the correct Python version:

    ```bash
    pyenv install --skip-existing
    ```

3. Install the dependencies:

    ```bash
    pdm install
    pdm run python -m ensurepip
    ```

4. Install the pre-commit hooks:

    ```bash
    pdm run pre-commit install
    ```

5. Create `.env` file by copying the `.env.template` file and updating the values as needed:

    ```bash
    cp .env.template .env
    ```

## Getting Started

### Code quality checks

Run the following command to check the code quality:

```bash
pdm run pre-commit run --all-files
pdm run pyre check
```

### Running the tests

Run the following command to run the tests:

```bash
pdm run tests
```

### Parsing the GDPR Handbook

Download GDPR handbook from [here](https://atpdk.sharepoint.com/sites/Fagligt/Shared%20Documents/Juridiske-retningslinjer/Persondata%20og%20GDPR/Haandbog%20for%20behandling%20af%20personoplysninger%20i%20ATP%20Koncernen_med%20ny%20layout.docx?web=1) and save it as `handbook.docx` in the `data/gdpr-handbook/raw` directory.

Then run:

```bash
pdm run convert-word-to-markdown \
    --file-path data/gdpr-handbook/raw/handbook.docx \
    --output-dir data/gdpr-handbook/processed
```

The Markdown version of the handbook can be parsed using the following command:

```bash
pdm run parse-markdown --input-path data/gdpr-handbook/processed/handbook.md
```

### Creating synthetic questions

Run the following command to create synthetic questions:

```bash
pdm run create-questions \
    --input-file-path data/gdpr-handbook/processed/handbook-cleaned.json \
    --output-dir data/gdpr-handbook/processed \
    --n-questions-per-section 5
```

### Creating reference answers

Run the following command to create reference answers:

```bash
pdm run create-reference-answers \
    --markdown-file-path data/gdpr-handbook/processed/handbook-cleaned.json \
    --questions-file-path data/gdpr-handbook/processed/handbook-cleaned-questions.json \
    --n-answers 1
```

### Chunking the text

To chunk the handbook using the paragraph method, run the following command:

```bash
pdm run chunk \
    --input-file-path data/gdpr-handbook/processed/handbook-cleaned.json \
    --max-chunk-size 128 \
    --chunking-method paragraph \
    --output-file-path data/gdpr-handbook/processed/handbook-cleaned-chunked-paragraph.json
```

To chunk the handbook using the naive method, run the following command:

```bash
pdm run chunk \
    --input-file-path data/gdpr-handbook/processed/handbook-cleaned.json \
    --max-chunk-size 128 \
    --chunking-method naive \
    --output-file-path data/gdpr-handbook/processed/handbook-cleaned-chunked-naive.json
```

### Mapping the questions to the chunks

To map the questions to the chunks, run the following command:

```bash
 pdm run map-questions-to-chunks \
    --questions-file-path data/gdpr-handbook/processed/generated-questions-for-handbook-cleaned.json \
    --chunk-file-path data/gdpr-handbook/processed/handbook-cleaned-chunked-paragraph.json \
    --output-path data/gdpr-handbook/processed/handbook-cleaned-questions-with-chunked-paragraph.json
```

## Creating the retriever index

To create the BM25 index, run the following command:

```bash
pdm run create-bm25-index \
    --data-file-path data/gdpr-handbook/processed/handbook-cleaned-chunked-paragraph.json \
    --storage-dir data/gdpr-handbook/processed/handbook-cleaned-chunked-paragraph-bm25
```

To create the vector index, run the following command:

```bash
pdm run create-vector-index \
    --data-file-path data/gdpr-handbook/processed/handbook-cleaned-chunked-paragraph.json \
    --storage-dir data/gdpr-handbook/processed/handbook-cleaned-chunked-paragraph-vector-index
```

## Evaluating retrieval performance

To evaluate the retrieval performance for BM25, run the following command:

```bash
pdm run eval-retriever \
    --annotation-set-file-path data/gdpr-handbook/processed/handbook-cleaned-questions-with-chunked-paragraph.json \
    --index-dir data/gdpr-handbook/processed/handbook-cleaned-chunked-paragraph-bm25 \
    --output-path data/gdpr-handbook/processed/eval-chunked-paragraph-bm25.json
```

To evaluate the retrieval performance for Vector Search, run the following command:

```bash
pdm run eval-retriever \
    --annotation-set-file-path data/gdpr-handbook/processed/handbook-cleaned-questions-with-chunked-paragraph.json \
    --index-dir data/gdpr-handbook/processed/handbook-cleaned-chunked-paragraph-vector-index \
    --output-path data/gdpr-handbook/processed/eval-chunked-paragraph-vector.json
```
