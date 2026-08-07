import { http, HttpResponse } from "msw";
import { screen } from "@testing-library/react";
import DocumentsScreen from "@/screens/documents/DocumentsScreen";
import { makeCard } from "@/test/factories";
import { renderWithClient } from "@/test/render";
import { server } from "@/test/server";

function mockApi(cards = [makeCard()], jobs: unknown[] = []) {
  server.use(
    http.get("/documents", () => HttpResponse.json(cards)),
    http.get("/jobs", () => HttpResponse.json(jobs)),
  );
}

test("каталог: строка с метаданными", async () => {
  mockApi([makeCard({
    title: "Чистая архитектура", author: "Роберт Мартин",
    topic: "Дизайн", tags: ["arch", "solid"], chunks: 214,
  })]);
  renderWithClient(<DocumentsScreen />);
  expect(await screen.findByText("Чистая архитектура")).toBeInTheDocument();
  expect(screen.getByText("Роберт Мартин")).toBeInTheDocument();
  expect(screen.getByText("Дизайн")).toBeInTheDocument();
  expect(screen.getByText("arch")).toBeInTheDocument();
  expect(screen.getByText("214")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Редактировать" })).toBeInTheDocument();
});

test("без автора и темы — прочерки", async () => {
  mockApi([makeCard({ author: null, topic: null })]);
  renderWithClient(<DocumentsScreen />);
  expect(await screen.findByText("Книга")).toBeInTheDocument();
  expect(screen.getAllByText("—")).toHaveLength(2); // автор и тема
});

test("без title показывается basename файла", async () => {
  mockApi([makeCard({ title: null, source_file: "/uploads/notes.md" })]);
  renderWithClient(<DocumentsScreen />);
  expect(await screen.findByText("notes.md")).toBeInTheDocument();
});

test("пустой каталог — явное состояние", async () => {
  mockApi([]);
  renderWithClient(<DocumentsScreen />);
  expect(await screen.findByText(/хранилище пустое/i)).toBeInTheDocument();
});
