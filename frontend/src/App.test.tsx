import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "@/App";
import { renderWithClient } from "@/test/render";

test("сайдбар: три раздела, переключение видимости", async () => {
  renderWithClient(<App />);
  const user = userEvent.setup();
  // по умолчанию активен Чат
  expect(screen.getByRole("button", { name: /чат/i })).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: /документы/i }));
  expect(screen.getByTestId("screen-documents")).toBeVisible();
  expect(screen.getByTestId("screen-chat")).not.toBeVisible();
});

test("неактивные экраны остаются смонтированными (состояние не теряется)", async () => {
  renderWithClient(<App />);
  const user = userEvent.setup();
  await user.click(screen.getByRole("button", { name: /поиск/i }));
  // чат скрыт, но НЕ размонтирован
  expect(screen.getByTestId("screen-chat")).toBeInTheDocument();
});
