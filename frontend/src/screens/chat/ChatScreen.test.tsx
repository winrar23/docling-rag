import { delay, http, HttpResponse } from "msw";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import ChatScreen from "@/screens/chat/ChatScreen";
import { renderWithClient } from "@/test/render";
import { server } from "@/test/server";

const ANSWER = {
  answer: "Data Vault — методология DWH.",
  sources: [{ file: "dwh-book.pdf", page: 87, headings: ["Data Vault"], score: 0.82 }],
};

test("отправка: ответ и чипы источников; history пуста на первом ходе", async () => {
  let body: unknown = null;
  server.use(
    http.post("/chat", async ({ request }) => {
      body = await request.json();
      return HttpResponse.json(ANSWER);
    }),
  );
  renderWithClient(<ChatScreen />);
  const user = userEvent.setup();
  await user.type(screen.getByPlaceholderText(/спросить/i), "Что такое Data Vault?");
  await user.click(screen.getByRole("button", { name: /отправить/i }));
  expect(await screen.findByText(/методология DWH/)).toBeInTheDocument();
  expect(screen.getByText(/dwh-book\.pdf · стр\. 87/)).toBeInTheDocument();
  expect(body).toEqual({ message: "Что такое Data Vault?", history: [] });
});

test("второй ход уносит историю из двух реплик", async () => {
  const bodies: unknown[] = [];
  server.use(
    http.post("/chat", async ({ request }) => {
      bodies.push(await request.json());
      return HttpResponse.json(ANSWER);
    }),
  );
  renderWithClient(<ChatScreen />);
  const user = userEvent.setup();
  const input = screen.getByPlaceholderText(/спросить/i);
  await user.type(input, "Вопрос 1");
  await user.click(screen.getByRole("button", { name: /отправить/i }));
  await screen.findAllByText(/методология DWH/);
  await user.type(input, "Вопрос 2");
  await user.click(screen.getByRole("button", { name: /отправить/i }));
  await waitFor(() => expect(bodies).toHaveLength(2));
  expect(bodies[1]).toMatchObject({
    message: "Вопрос 2",
    history: [
      { role: "user", content: "Вопрос 1" },
      { role: "assistant", content: "Data Vault — методология DWH." },
    ],
  });
});

test("ошибка LLM: сообщение откатывается, текст возвращается в поле", async () => {
  server.use(
    http.post("/chat", () =>
      HttpResponse.json({ detail: "LLM не ответила за 120 секунд" }, { status: 504 }),
    ),
  );
  renderWithClient(<ChatScreen />);
  const user = userEvent.setup();
  await user.type(screen.getByPlaceholderText(/спросить/i), "Вопрос");
  await user.click(screen.getByRole("button", { name: /отправить/i }));
  expect(await screen.findByText("LLM не ответила за 120 секунд")).toBeInTheDocument(); // toast
  expect(screen.getByPlaceholderText(/спросить/i)).toHaveValue("Вопрос"); // текст возвращён
  expect(screen.queryByText(/^Вопрос$/)).not.toBeInTheDocument(); // из ленты откачен
});

test("«Новый диалог» очищает ленту", async () => {
  server.use(http.post("/chat", () => HttpResponse.json(ANSWER)));
  renderWithClient(<ChatScreen />);
  const user = userEvent.setup();
  await user.type(screen.getByPlaceholderText(/спросить/i), "Вопрос");
  await user.click(screen.getByRole("button", { name: /отправить/i }));
  await screen.findByText(/методология DWH/);
  await user.click(screen.getByRole("button", { name: /новый диалог/i }));
  expect(screen.queryByText(/методология DWH/)).not.toBeInTheDocument();
});

test("«Новый диалог» задизейблен, пока запрос в полёте", async () => {
  server.use(
    http.post("/chat", async () => {
      await delay(150);
      return HttpResponse.json(ANSWER);
    }),
  );
  renderWithClient(<ChatScreen />);
  const user = userEvent.setup();
  await user.type(screen.getByPlaceholderText(/спросить/i), "Вопрос");
  await user.click(screen.getByRole("button", { name: /отправить/i }));
  expect(screen.getByRole("button", { name: /новый диалог/i })).toBeDisabled();
  await screen.findByText(/методология DWH/);
  expect(screen.getByRole("button", { name: /новый диалог/i })).toBeEnabled();
});
