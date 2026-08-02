// @vitest-environment happy-dom
import { http, HttpResponse } from "msw";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import UploadDialog from "@/screens/documents/UploadDialog";
import { renderWithClient } from "@/test/render";
import { server } from "@/test/server";

test("отправляет multipart со всеми form-полями контракта", async () => {
  let form: FormData | null = null;
  server.use(
    http.post("/documents", async ({ request }) => {
      form = await request.formData();
      return HttpResponse.json({ job_id: "j1", status: "queued" }, { status: 202 });
    }),
  );
  renderWithClient(<UploadDialog />);
  const user = userEvent.setup();
  await user.click(screen.getByRole("button", { name: /загрузить документ/i }));
  await user.upload(
    // ASCII-имя файла: happy-dom мохибейкает non-ASCII filename в multipart
    // Content-Disposition (см. отчёт задачи) — кириллица проверяется в title/topic/tags,
    // это и есть реальный сценарий (OCR-метаданные), а не имя файла
    screen.getByLabelText(/файл/i),
    new File(["%PDF"], "scan.pdf", { type: "application/pdf" }),
  );
  await user.type(screen.getByLabelText(/название/i), "Русский скан");
  await user.type(screen.getByLabelText(/тема/i), "OCR");
  await user.type(screen.getByLabelText(/теги/i), "ocr, ru");
  await user.selectOptions(screen.getByLabelText(/режим ocr/i), "on");
  await user.selectOptions(screen.getByLabelText(/язык ocr/i), "ru");
  await user.click(screen.getByRole("button", { name: /^загрузить$/i }));

  await waitFor(() => expect(form).not.toBeNull());
  expect((form!.get("file") as File).name).toBe("scan.pdf");
  expect(form!.get("title")).toBe("Русский скан");
  expect(form!.get("topic")).toBe("OCR");
  expect(form!.getAll("tags")).toEqual(["ocr", "ru"]);
  expect(form!.get("ocr")).toBe("on");
  expect(form!.get("ocr_lang")).toBe("ru");
});

test("409 (дубль) показывает сообщение бэкенда", async () => {
  server.use(
    http.post("/documents", () =>
      HttpResponse.json({ detail: { message: "Уже индексируется", job_id: "j0" } }, { status: 409 }),
    ),
  );
  renderWithClient(<UploadDialog />);
  const user = userEvent.setup();
  await user.click(screen.getByRole("button", { name: /загрузить документ/i }));
  await user.upload(screen.getByLabelText(/файл/i), new File(["x"], "a.pdf"));
  await user.click(screen.getByRole("button", { name: /^загрузить$/i }));
  expect(await screen.findByText("Уже индексируется")).toBeInTheDocument(); // sonner toast
});
