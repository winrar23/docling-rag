import { delay, http, HttpResponse } from "msw";
import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import ChatScreen, { splitParagraphs } from "@/screens/chat/ChatScreen";
import { renderWithClient } from "@/test/render";
import { server } from "@/test/server";

const ANSWER = {
  answer: "Data Vault — методология DWH.",
  sources: [
    {
      file: "dwh-book.pdf",
      page: 87,
      headings: ["Data Vault"],
      score: 0.82,
      text: "Data Vault 2.0 состоит из хабов, линков и сателлитов.",
    },
  ],
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

test("«Очистить диалог» очищает ленту", async () => {
  server.use(http.post("/chat", () => HttpResponse.json(ANSWER)));
  renderWithClient(<ChatScreen />);
  const user = userEvent.setup();
  await user.type(screen.getByPlaceholderText(/спросить/i), "Вопрос");
  await user.click(screen.getByRole("button", { name: /отправить/i }));
  await screen.findByText(/методология DWH/);
  await user.click(screen.getByRole("button", { name: /очистить диалог/i }));
  expect(screen.queryByText(/методология DWH/)).not.toBeInTheDocument();
});

test("«Очистить диалог» задизейблен, пока запрос в полёте", async () => {
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
  expect(screen.getByRole("button", { name: /очистить диалог/i })).toBeDisabled();
  await screen.findByText(/методология DWH/);
  expect(screen.getByRole("button", { name: /очистить диалог/i })).toBeEnabled();
});

const TWO_SOURCES = {
  answer: "Ответ с двумя источниками.",
  sources: [
    { file: "dwh-book.pdf", page: 87, headings: ["Data Vault"], score: 0.82, text: "Фрагмент про Data Vault." },
    { file: "arch-book.pdf", page: 12, headings: ["Архитектура"], score: 0.75, text: "Фрагмент про архитектуру." },
  ],
};

const NO_TEXT = {
  answer: "Ответ от старого бэкенда.",
  sources: [{ file: "old-api.pdf", page: 3, headings: [], score: 0.5 }],
};

async function sendQuestion(user: ReturnType<typeof userEvent.setup>) {
  await user.type(screen.getByPlaceholderText(/спросить/i), "Вопрос");
  await user.click(screen.getByRole("button", { name: /отправить/i }));
}

test("клик по чипу открывает панель с текстом фрагмента и метаданными", async () => {
  server.use(http.post("/chat", () => HttpResponse.json(ANSWER)));
  renderWithClient(<ChatScreen />);
  const user = userEvent.setup();
  await sendQuestion(user);
  await user.click(await screen.findByText(/dwh-book\.pdf · стр\. 87/));
  const panel = screen.getByTestId("source-panel");
  expect(panel.className).toContain("translate-x-0");
  expect(within(panel).getByText(/хабов, линков и сателлитов/)).toBeInTheDocument();
  expect(within(panel).getByText("Data Vault")).toBeInTheDocument(); // headings-цепочка
  expect(within(panel).getByText(/стр\. 87/)).toBeInTheDocument();
  expect(within(panel).getByText(/score 0\.82/)).toBeInTheDocument();
});

test("клик по области чата закрывает панель", async () => {
  server.use(http.post("/chat", () => HttpResponse.json(ANSWER)));
  renderWithClient(<ChatScreen />);
  const user = userEvent.setup();
  await sendQuestion(user);
  await user.click(await screen.findByText(/dwh-book\.pdf · стр\. 87/));
  expect(screen.getByTestId("source-panel").className).toContain("translate-x-0");
  await user.click(screen.getByRole("heading", { name: "Чат" })); // клик «мимо» — по шапке экрана
  expect(screen.getByTestId("source-panel").className).toContain("translate-x-full");
});

test("клик по чипу при открытой панели не закрывает её (stopPropagation)", async () => {
  server.use(http.post("/chat", () => HttpResponse.json(ANSWER)));
  renderWithClient(<ChatScreen />);
  const user = userEvent.setup();
  await sendQuestion(user);
  const chip = await screen.findByText(/dwh-book\.pdf · стр\. 87/);
  await user.click(chip);
  await user.click(chip); // повторный клик по тому же чипу
  expect(screen.getByTestId("source-panel").className).toContain("translate-x-0");
});

test("клик по другому чипу подменяет контент панели после анимации", async () => {
  server.use(http.post("/chat", () => HttpResponse.json(TWO_SOURCES)));
  renderWithClient(<ChatScreen />);
  const user = userEvent.setup();
  await sendQuestion(user);
  await user.click(await screen.findByText(/dwh-book\.pdf · стр\. 87/));
  const panel = screen.getByTestId("source-panel");
  expect(within(panel).getByText(/про Data Vault/)).toBeInTheDocument();
  await user.click(screen.getByText(/arch-book\.pdf · стр\. 12/));
  // подмена происходит после выезда (таймер 200 мс) — ждём реальными таймерами
  await waitFor(() => {
    expect(within(panel).getByText(/про архитектуру/)).toBeInTheDocument();
  });
  expect(panel.className).toContain("translate-x-0");
});

test("источник без text показывает фолбэк-текст", async () => {
  server.use(http.post("/chat", () => HttpResponse.json(NO_TEXT)));
  renderWithClient(<ChatScreen />);
  const user = userEvent.setup();
  await sendQuestion(user);
  await user.click(await screen.findByText(/old-api\.pdf · стр\. 3/));
  expect(
    within(screen.getByTestId("source-panel")).getByText(/Текст фрагмента недоступен/),
  ).toBeInTheDocument();
});

test("splitParagraphs: граница абзаца после конца предложения", () => {
  expect(splitParagraphs("А кончилось.\nНовое началось")).toEqual(["А кончилось.", "Новое началось"]);
});

test("splitParagraphs: PDF-перенос без терминатора склеивается пробелом", () => {
  expect(splitParagraphs("руководства\nданными, дальше.")).toEqual(["руководства данными, дальше."]);
});

test("splitParagraphs: текст без переносов — один абзац", () => {
  expect(splitParagraphs("Один абзац без переносов.")).toEqual(["Один абзац без переносов."]);
});

test("splitParagraphs: терминатор с закрывающей кавычкой — граница", () => {
  expect(splitParagraphs("Кончилось!»\nНовое")).toEqual(["Кончилось!»", "Новое"]);
});

test("splitParagraphs: пустые фрагменты отбрасываются", () => {
  expect(splitParagraphs("а.\n\nб")).toEqual(["а.", "б"]);
});

test("splitParagraphs: пустая строка — пустой список", () => {
  expect(splitParagraphs("")).toEqual([]);
});

const MULTI_PARA = {
  answer: "Ответ с многоабзацным источником.",
  sources: [
    {
      file: "book.pdf", page: 5, headings: [], score: 0.9, element_type: "text",
      text: "Первый абзац закончился.\nВторой абзац начался",
    },
  ],
};

const GLUED = {
  answer: "Ответ со склейкой.",
  sources: [
    {
      file: "book.pdf", page: 6, headings: [], score: 0.9, element_type: "text",
      text: "руководства\nданными в организации.",
    },
  ],
};

const TABLE_SOURCE = {
  answer: "Ответ с таблицей.",
  sources: [
    {
      file: "book.pdf", page: 7, headings: [], score: 0.9, element_type: "table",
      text: "Колонка = значение.\nКолонка2 = значение2.",
    },
  ],
};

test("панель: текст источника рендерится отдельными абзацами", async () => {
  server.use(http.post("/chat", () => HttpResponse.json(MULTI_PARA)));
  renderWithClient(<ChatScreen />);
  const user = userEvent.setup();
  await sendQuestion(user);
  await user.click(await screen.findByText(/book\.pdf · стр\. 5/));
  const panel = screen.getByTestId("source-panel");
  // точные getByText сработают только если каждый абзац — отдельный элемент
  const first = within(panel).getByText("Первый абзац закончился.");
  const second = within(panel).getByText("Второй абзац начался");
  expect(first.tagName).toBe("P");
  expect(second.tagName).toBe("P");
  expect(first).not.toBe(second);
});

test("панель: PDF-перенос склеен в один абзац", async () => {
  server.use(http.post("/chat", () => HttpResponse.json(GLUED)));
  renderWithClient(<ChatScreen />);
  const user = userEvent.setup();
  await sendQuestion(user);
  await user.click(await screen.findByText(/book\.pdf · стр\. 6/));
  const panel = screen.getByTestId("source-panel");
  expect(within(panel).getByText("руководства данными в организации.")).toBeInTheDocument();
});

test("панель: table-источник сохраняет переносы (pre-wrap)", async () => {
  server.use(http.post("/chat", () => HttpResponse.json(TABLE_SOURCE)));
  renderWithClient(<ChatScreen />);
  const user = userEvent.setup();
  await sendQuestion(user);
  await user.click(await screen.findByText(/book\.pdf · стр\. 7/));
  const panel = screen.getByTestId("source-panel");
  const el = within(panel).getByText(/Колонка = значение\./);
  expect(el.className).toContain("whitespace-pre-wrap");
  expect(el.textContent).toContain("\n");
});
