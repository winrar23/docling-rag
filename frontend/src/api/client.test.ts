import { http, HttpResponse } from "msw";
import { api, ApiError, basename, detailMessage } from "@/api/client";
import { server } from "@/test/server";
import { makeCard } from "@/test/factories";

test("listDocuments возвращает карточки", async () => {
  server.use(http.get("/documents", () => HttpResponse.json([makeCard()])));
  const cards = await api.listDocuments();
  expect(cards).toHaveLength(1);
  expect(cards[0].title).toBe("Книга");
});

test("не-2xx превращается в ApiError со status и detail", async () => {
  server.use(
    http.get("/documents", () =>
      HttpResponse.json({ detail: "PostgreSQL недоступен: ..." }, { status: 503 }),
    ),
  );
  const err = await api.listDocuments().catch((e) => e);
  expect(err).toBeInstanceOf(ApiError);
  expect(err.status).toBe(503);
  expect(detailMessage(err)).toBe("PostgreSQL недоступен: ...");
});

test("detail-объект (409) даёт message", () => {
  const err = new ApiError(409, { message: "Уже индексируется", job_id: "j1" });
  expect(detailMessage(err)).toBe("Уже индексируется");
});

test("search собирает query-параметры контракта", async () => {
  let url = "";
  server.use(
    http.get("/search", ({ request }) => {
      url = request.url;
      return HttpResponse.json({ query: "q", results: [] });
    }),
  );
  await api.search({ q: "паттерны", topK: 7, tag: "arch", topic: "DWH" });
  const qs = new URL(url).searchParams;
  expect(qs.get("q")).toBe("паттерны");
  expect(qs.get("top_k")).toBe("7");
  expect(qs.get("tag")).toBe("arch");
  expect(qs.get("topic")).toBe("DWH");
});

test("basename отрезает путь", () => {
  expect(basename("/uploads/книга.pdf")).toBe("книга.pdf");
  expect(basename("book.md")).toBe("book.md");
});

test("422 array-detail (FastAPI-валидация) собирается из msg через «; »", () => {
  const err = new ApiError(422, [
    { loc: ["query", "top_k"], msg: "Input should be greater than or equal to 1", type: "greater_than_equal" },
    { loc: ["query", "q"], msg: "Field required", type: "missing" },
  ]);
  expect(detailMessage(err)).toBe("Input should be greater than or equal to 1; Field required");
});

test("мусорный array-detail без msg падает в фолбэк «Ошибка N»", () => {
  const err = new ApiError(422, [{ unexpected: true }, "строка"]);
  expect(detailMessage(err)).toBe("Ошибка 422");
});
