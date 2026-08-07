import { http, HttpResponse } from "msw";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import SearchScreen from "@/screens/search/SearchScreen";
import { makeCard } from "@/test/factories";
import { renderWithClient } from "@/test/render";
import { server } from "@/test/server";

const RESULT = {
  text: "Хабы содержат бизнес-ключи…",
  score: 0.82,
  source_file: "/uploads/dwh-book.pdf",
  title: "DWH",
  page_number: 87,
  headings: ["Data Vault", "Hubs"],
  element_type: "text",
};

function mockApi(results: unknown[] = [RESULT]) {
  let url = "";
  server.use(
    http.get("/documents", () => HttpResponse.json([makeCard({ tags: ["dwh"], topic: "Данные" })])),
    http.get("/search", ({ request }) => {
      url = request.url;
      return HttpResponse.json({ query: "q", results });
    }),
  );
  return () => url;
}

test("сабмит уходит с параметрами, результат рендерится", async () => {
  const getUrl = mockApi();
  renderWithClient(<SearchScreen />);
  const user = userEvent.setup();
  await user.type(screen.getByPlaceholderText(/поисковый запрос/i), "data vault");
  await user.selectOptions(await screen.findByLabelText(/тег/i), "dwh");
  await user.click(screen.getByRole("button", { name: /найти/i }));
  expect(await screen.findByText(/Хабы содержат/)).toBeInTheDocument();
  expect(screen.getByText(/dwh-book\.pdf/)).toBeInTheDocument();
  expect(screen.getByText(/стр\. 87/)).toBeInTheDocument();
  expect(screen.getByText(/Data Vault → Hubs/)).toBeInTheDocument();
  expect(screen.getByText("0.82")).toBeInTheDocument();
  const qs = new URL(getUrl()).searchParams;
  expect(qs.get("q")).toBe("data vault");
  expect(qs.get("tag")).toBe("dwh");
});

test("пустой результат — «Ничего не найдено»", async () => {
  mockApi([]);
  renderWithClient(<SearchScreen />);
  const user = userEvent.setup();
  await user.type(screen.getByPlaceholderText(/поисковый запрос/i), "ничто");
  await user.click(screen.getByRole("button", { name: /найти/i }));
  expect(await screen.findByText(/ничего не найдено/i)).toBeInTheDocument();
});

test("бейдж element_type для таблиц", async () => {
  mockApi([{ ...RESULT, element_type: "table" }]);
  renderWithClient(<SearchScreen />);
  const user = userEvent.setup();
  await user.type(screen.getByPlaceholderText(/поисковый запрос/i), "таблица");
  await user.click(screen.getByRole("button", { name: /найти/i }));
  expect(await screen.findByText("table")).toBeInTheDocument();
});

test("возврат фокуса в окно не перезапрашивает поиск (не спамит searches)", async () => {
  let calls = 0;
  server.use(
    http.get("/documents", () => HttpResponse.json([makeCard({ tags: ["dwh"], topic: "Данные" })])),
    http.get("/search", () => {
      calls += 1;
      return HttpResponse.json({ query: "q", results: [RESULT] });
    }),
  );
  renderWithClient(<SearchScreen />);
  const user = userEvent.setup();
  await user.type(screen.getByPlaceholderText(/поисковый запрос/i), "data vault");
  await user.click(screen.getByRole("button", { name: /найти/i }));
  await screen.findByText(/Хабы содержат/);
  // TanStack v5 focusManager слушает visibilitychange на window
  window.dispatchEvent(new Event("visibilitychange"));
  await new Promise((r) => setTimeout(r, 50)); // даём refetch'у шанс уйти
  expect(calls).toBe(1);
});

test("повторный сабмит идентичного запроса перезапрашивает", async () => {
  let calls = 0;
  server.use(
    http.get("/documents", () => HttpResponse.json([makeCard({ tags: ["dwh"], topic: "Данные" })])),
    http.get("/search", () => {
      calls += 1;
      return HttpResponse.json({ query: "q", results: [RESULT] });
    }),
  );
  renderWithClient(<SearchScreen />);
  const user = userEvent.setup();
  await user.type(screen.getByPlaceholderText(/поисковый запрос/i), "data vault");
  await user.click(screen.getByRole("button", { name: /найти/i }));
  await screen.findByText(/Хабы содержат/);
  await user.click(screen.getByRole("button", { name: /найти/i }));
  await waitFor(() => expect(calls).toBe(2));
});

test("результаты группируются по документу: имя файла один раз, бейджи с числом фрагментов", async () => {
  mockApi([
    RESULT,
    { ...RESULT, text: "Сателлиты хранят атрибуты…", page_number: 91 },
    { ...RESULT, source_file: "/uploads/other.pdf", text: "Линки связывают хабы…" },
  ]);
  renderWithClient(<SearchScreen />);
  const user = userEvent.setup();
  await user.type(screen.getByPlaceholderText(/поисковый запрос/i), "data vault");
  await user.click(screen.getByRole("button", { name: /найти/i }));
  await screen.findByText(/Хабы содержат/);
  // имя файла — один раз (заголовок группы), а не в каждой карточке
  expect(screen.getAllByText(/dwh-book\.pdf/)).toHaveLength(1);
  // бейджи групп: 2 фрагмента у dwh-book, 1 фрагмент у other
  expect(screen.getByText("2 фрагмента")).toBeInTheDocument();
  expect(screen.getByText("1 фрагмент")).toBeInTheDocument();
});

test("счётчик найденного: «N фрагментов» с правильным плюралом", async () => {
  mockApi([
    RESULT,
    { ...RESULT, text: "Сателлиты хранят атрибуты…", page_number: 91 },
    { ...RESULT, source_file: "/uploads/other.pdf", text: "Линки связывают хабы…" },
  ]);
  renderWithClient(<SearchScreen />);
  const user = userEvent.setup();
  await user.type(screen.getByPlaceholderText(/поисковый запрос/i), "data vault");
  await user.click(screen.getByRole("button", { name: /найти/i }));
  await screen.findByText(/Хабы содержат/);
  // общий счётчик (3 = «фрагмента»); уникален: бейджи групп — «2 фрагмента» и «1 фрагмент»
  expect(screen.getByText("3 фрагмента")).toBeInTheDocument();
});
