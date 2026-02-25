# Baseline Test Scenarios for docling-rag-manager

These scenarios test whether a Claude Code agent can manage docling-rag correctly without any skill guidance.

---

## Scenario 1: Full Pipeline (bootstrap + index + search)

IMPORTANT: This is a real task. Execute it, don't just describe what you'd do.

You are in the directory /Users/danny/Documents/Документы Даниил/Github/docling-rag-dev.
There are two PDF files in the root:
- Building-a-scalable-data-warehouse-with-data-vault-20_part_1.pdf
- Building-a-scalable-data-warehouse-with-data-vault-20_part_2.pdf

Task: Index both PDFs with topic "data vault" and tag "architecture", then search for "hub table design patterns".

Do not read CLAUDE.md or any project documentation. Just use the tools available to you.

---

## Scenario 2: List + Status Check

IMPORTANT: This is a real task. Execute it.

You are in /Users/danny/Documents/Документы Даниил/Github/docling-rag-dev.
Check what documents are already indexed and show their metadata (title, topic, tags, chunk counts).

Do not read CLAUDE.md or any project documentation.

---

## Scenario 3: Add with Metadata

IMPORTANT: This is a real task. Execute it.

You are in /Users/danny/Documents/Документы Даниил/Github/docling-rag-dev.
There is a directory `data/` that may or may not be initialized.
Add the file `Building-a-scalable-data-warehouse-with-data-vault-20_part_1.pdf` with:
- title: "Data Vault 2.0 Part 1"
- topic: "Data Warehousing"
- tags: architecture, data-vault

Do not read CLAUDE.md or any project documentation.
