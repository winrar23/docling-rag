export type JobStatus = "queued" | "running" | "done" | "failed";

// GET /documents (элемент списка) == GET /documents/{id} — карточка одна и та же
export interface DocumentCard {
  id: string;
  source_file: string;
  title: string | null;
  author: string | null;
  topic: string | null;
  tags: string[];
  added_at: string; // ISO
  chunks: number;
  indexing: { status: JobStatus; job_id: string } | null;
}

// PATCH /documents/{id} — диалог правки шлёт все четыре поля (полная правка карточки)
export interface DocumentPatchBody {
  title: string | null;
  author: string | null;
  topic: string | null;
  tags: string[];
}

// GET /jobs, GET /jobs/{id} (_COLS из db_jobs.py + elapsed/heartbeat из _with_liveness)
export interface Job {
  id: string;
  source_file: string;
  original_name: string;
  status: JobStatus;
  step: string | null;
  chunks_total: number | null;
  chunks_done: number | null;
  error: string | null;
  warning: string | null; // не-фатальный сбой шага metadata
  attempts: number;
  created_at: string;
  started_at: string | null;
  updated_at: string | null;
  finished_at: string | null;
  ocr: "auto" | "on" | "off";
  ocr_lang: "en" | "ru";
  elapsed_sec: number | null;
  heartbeat_age_sec: number | null;
}

export interface SearchResult {
  text: string;
  score: number;
  source_file: string;
  title: string | null;
  page_number: number | null;
  headings: string[];
  element_type: string;
}

export interface SearchResponse {
  query: string;
  results: SearchResult[];
}

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

export interface ChatSource {
  file: string; // уже basename (так отдаёт бэкенд)
  page: number;
  headings: string[];
  score: number;
  text?: string; // текст фрагмента для панели источника; опционален — старый бэкенд его не отдаёт
  element_type?: string; // text | table | code; опционален — старый бэкенд не отдаёт, undefined = обычный текст
}

export interface ChatResponse {
  answer: string;
  sources: ChatSource[];
}

export interface UploadAccepted {
  job_id: string;
  status: "queued";
}

export interface DeleteResponse {
  deleted: string;
  chunks: number;
  file_removed: boolean;
}
