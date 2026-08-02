import type { DocumentCard, Job } from "@/api/types";

export function makeCard(over: Partial<DocumentCard> = {}): DocumentCard {
  return {
    id: "11111111-1111-1111-1111-111111111111",
    source_file: "/uploads/book.pdf",
    title: "Книга",
    topic: "Архитектура",
    tags: ["arch"],
    added_at: "2026-08-01T10:00:00+00:00",
    chunks: 42,
    indexing: null,
    ...over,
  };
}

export function makeJob(over: Partial<Job> = {}): Job {
  return {
    id: "22222222-2222-2222-2222-222222222222",
    source_file: "/uploads/book.pdf",
    original_name: "book.pdf",
    title: null,
    topic: null,
    tags: [],
    status: "running",
    step: "чанкинг",
    chunks_total: 120,
    chunks_done: 43,
    error: null,
    attempts: 1,
    created_at: "2026-08-01T10:00:00+00:00",
    started_at: "2026-08-01T10:00:05+00:00",
    updated_at: "2026-08-01T10:00:30+00:00",
    finished_at: null,
    ocr: "auto",
    ocr_lang: "en",
    elapsed_sec: 30,
    heartbeat_age_sec: 2,
  };
}
