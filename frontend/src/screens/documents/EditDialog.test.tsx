import { http, HttpResponse } from "msw";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import EditDialog from "@/screens/documents/EditDialog";
import { makeCard } from "@/test/factories";
import { renderWithClient } from "@/test/render";
import { server } from "@/test/server";

const card = makeCard({
  id: "d1", title: "Книга", author: "Автор А.", topic: "бд", tags: ["postgres"],
});

test("префиллит поля и шлёт PATCH", async () => {
  let sent: unknown = null;
  server.use(
    http.patch("/documents/d1", async ({ request }) => {
      sent = await request.json();
      return HttpResponse.json({ ...card, title: "Новая" });
    }),
  );
  renderWithClient(<EditDialog card={card} />);
  const user = userEvent.setup();
  await user.click(screen.getByRole("button", { name: "Редактировать" }));
  const title = screen.getByLabelText("Название");
  expect(title).toHaveValue("Книга");
  expect(screen.getByLabelText("Автор")).toHaveValue("Автор А.");
  expect(screen.getByLabelText("Тема")).toHaveValue("бд");
  expect(screen.getByLabelText(/Теги/)).toHaveValue("postgres");
  await user.clear(title);
  await user.type(title, "Новая");
  await user.click(screen.getByRole("button", { name: "Сохранить" }));
  await waitFor(() => expect(sent).toEqual({
    title: "Новая", author: "Автор А.", topic: "бд", tags: ["postgres"],
  }));
  expect(await screen.findByText("Метаданные обновлены")).toBeInTheDocument();
});

test("пустые поля уходят null, пустые теги — []", async () => {
  let sent: unknown = null;
  server.use(
    http.patch("/documents/d1", async ({ request }) => {
      sent = await request.json();
      return HttpResponse.json({ ...card, author: null, topic: null, tags: [] });
    }),
  );
  renderWithClient(<EditDialog card={card} />);
  const user = userEvent.setup();
  await user.click(screen.getByRole("button", { name: "Редактировать" }));
  await user.clear(screen.getByLabelText("Автор"));
  await user.clear(screen.getByLabelText("Тема"));
  await user.clear(screen.getByLabelText(/Теги/));
  await user.click(screen.getByRole("button", { name: "Сохранить" }));
  await waitFor(() => expect(sent).toEqual({
    title: "Книга", author: null, topic: null, tags: [],
  }));
});

test("409 (идёт индексация) показывает сообщение бэкенда", async () => {
  server.use(
    http.patch("/documents/d1", () =>
      HttpResponse.json({ detail: { message: "Идёт индексация", job_id: "j1" } }, { status: 409 }),
    ),
  );
  renderWithClient(<EditDialog card={card} />);
  const user = userEvent.setup();
  await user.click(screen.getByRole("button", { name: "Редактировать" }));
  await user.click(screen.getByRole("button", { name: "Сохранить" }));
  expect(await screen.findByText("Идёт индексация")).toBeInTheDocument(); // sonner toast
});
