import { http, HttpResponse } from "msw";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import DocumentsScreen from "@/screens/documents/DocumentsScreen";
import { makeCard } from "@/test/factories";
import { renderWithClient } from "@/test/render";
import { server } from "@/test/server";

test("удаление: confirm → DELETE → строка исчезает, toast с итогом", async () => {
  let deleted = false;
  server.use(
    http.get("/jobs", () => HttpResponse.json([])),
    http.get("/documents", () => HttpResponse.json(deleted ? [] : [makeCard()])),
    http.delete("/documents/:id", () => {
      deleted = true;
      return HttpResponse.json({ deleted: "Книга", chunks: 42, file_removed: true });
    }),
  );
  renderWithClient(<DocumentsScreen />);
  const user = userEvent.setup();
  await user.click(await screen.findByRole("button", { name: /удалить/i }));
  await user.click(screen.getByRole("button", { name: /^да, удалить$/i })); // подтверждение в AlertDialog
  expect(await screen.findByText(/Удалено: Книга \(42 chunks\)/)).toBeInTheDocument();
  await waitFor(() => expect(screen.queryByText("Книга")).not.toBeInTheDocument());
});

test("409 при активной джобе — сообщение бэкенда", async () => {
  server.use(
    http.get("/jobs", () => HttpResponse.json([])),
    http.get("/documents", () => HttpResponse.json([makeCard()])),
    http.delete("/documents/:id", () =>
      HttpResponse.json({ detail: { message: "Идёт индексация", job_id: "j1" } }, { status: 409 }),
    ),
  );
  renderWithClient(<DocumentsScreen />);
  const user = userEvent.setup();
  await user.click(await screen.findByRole("button", { name: /удалить/i }));
  await user.click(screen.getByRole("button", { name: /^да, удалить$/i }));
  expect(await screen.findByText("Идёт индексация")).toBeInTheDocument();
});
