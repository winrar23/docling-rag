import { http, HttpResponse } from "msw";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach } from "vitest";
import App from "@/App";
import { renderWithClient } from "@/test/render";
import { server } from "@/test/server";

beforeEach(() => {
  server.use(
    http.get("/documents", () => HttpResponse.json([])),
    http.get("/jobs", () => HttpResponse.json([])),
  );
});

test("сайдбар: три раздела, переключение видимости", async () => {
  renderWithClient(<App />);
  const user = userEvent.setup();
  // по умолчанию активен Чат
  expect(screen.getByRole("button", { name: /чат/i })).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: /документы/i }));
  await waitFor(() => {
    expect(screen.getByTestId("screen-documents")).toBeVisible();
  });
  expect(screen.getByTestId("screen-chat")).not.toBeVisible();
});

test("неактивные экраны остаются смонтированными (состояние не теряется)", async () => {
  renderWithClient(<App />);
  const user = userEvent.setup();
  await user.click(screen.getByRole("button", { name: /поиск/i }));
  // чат скрыт, но НЕ размонтирован
  expect(screen.getByTestId("screen-chat")).toBeInTheDocument();
});

test("сайдбар: логотип Polka вместо docling-rag", () => {
  renderWithClient(<App />);
  expect(screen.getByText("Polka")).toBeInTheDocument();
  expect(screen.queryByText("docling-rag")).not.toBeInTheDocument();
});
