import { http, HttpResponse } from "msw";
import { screen } from "@testing-library/react";
import DocumentsScreen from "@/screens/documents/DocumentsScreen";
import { makeCard, makeJob } from "@/test/factories";
import { renderWithClient } from "@/test/render";
import { server } from "@/test/server";

function mockApi(jobs: unknown[], cards = [makeCard()]) {
  server.use(
    http.get("/documents", () => HttpResponse.json(cards)),
    http.get("/jobs", () => HttpResponse.json(jobs)),
  );
}

test("running-джоба: имя файла, шаг, прогресс", async () => {
  mockApi([makeJob({ status: "running", step: "чанкинг", chunks_done: 43, chunks_total: 120 })]);
  renderWithClient(<DocumentsScreen />);
  expect(await screen.findByText("book.pdf")).toBeInTheDocument();
  expect(screen.getByText(/чанкинг/)).toBeInTheDocument();
  expect(screen.getByText("43 / 120")).toBeInTheDocument();
});

test("failed-джоба: текст ошибки", async () => {
  mockApi([makeJob({ status: "failed", error: "Формат .txt не поддерживается" })]);
  renderWithClient(<DocumentsScreen />);
  expect(await screen.findByText(/Формат .txt не поддерживается/)).toBeInTheDocument();
});

test("done-джобы в секции не показываются", async () => {
  mockApi([makeJob({ status: "done", original_name: "готовая.pdf" })]);
  renderWithClient(<DocumentsScreen />);
  await screen.findByText("Книга"); // каталог отрисовался
  expect(screen.queryByText("готовая.pdf")).not.toBeInTheDocument();
});

test("queued-джоба без step показывает русский статус", async () => {
  mockApi([makeJob({ status: "queued", step: null, started_at: null, chunks_done: 0 })]);
  renderWithClient(<DocumentsScreen />);
  expect(await screen.findByText("в очереди")).toBeInTheDocument();
  expect(screen.queryByText("queued")).not.toBeInTheDocument();
});
