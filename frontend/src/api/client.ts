import type {
  ChatResponse,
  ChatTurn,
  DeleteResponse,
  DocumentCard,
  Job,
  SearchResponse,
  UploadAccepted,
} from "./types";

export class ApiError extends Error {
  constructor(
    public status: number,
    public detail: unknown,
  ) {
    super(typeof detail === "string" ? detail : JSON.stringify(detail));
    this.name = "ApiError";
  }
}

export function detailMessage(e: unknown): string {
  if (e instanceof ApiError) {
    if (typeof e.detail === "string") return e.detail;
    if (e.detail && typeof e.detail === "object" && "message" in e.detail) {
      return String((e.detail as { message: unknown }).message);
    }
    return `Ошибка ${e.status}`;
  }
  return "API недоступен";
}

export const basename = (p: string): string => p.split("/").pop() ?? p;

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    // new URL: относительный путь не принимается node-fetch'ем в тестах
    res = await fetch(new URL(path, window.location.origin), init);
  } catch {
    throw new ApiError(0, "API недоступен");
  }
  if (!res.ok) {
    let detail: unknown = `HTTP ${res.status}`;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* не-JSON тело — оставляем HTTP-код */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export interface SearchParams {
  q: string;
  topK?: number;
  tag?: string;
  topic?: string;
}

function searchQuery({ q, topK, tag, topic }: SearchParams): string {
  const qs = new URLSearchParams({ q });
  if (topK) qs.set("top_k", String(topK));
  if (tag) qs.append("tag", tag);
  if (topic) qs.set("topic", topic);
  return qs.toString();
}

export const api = {
  listDocuments: () => request<DocumentCard[]>("/documents"),
  deleteDocument: (id: string) =>
    request<DeleteResponse>(`/documents/${id}`, { method: "DELETE" }),
  listJobs: (limit = 100) => request<Job[]>(`/jobs?limit=${limit}`),
  search: (p: SearchParams) => request<SearchResponse>(`/search?${searchQuery(p)}`),
  chat: (message: string, history: ChatTurn[]) =>
    request<ChatResponse>("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, history }),
    }),
  upload: (form: FormData) => request<UploadAccepted>("/documents", { method: "POST", body: form }),
};
