import { splitParagraphs } from "@/lib/paragraphs";

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

test("splitParagraphs: дефисный PDF-перенос склеивается без пробела", () => {
  expect(splitParagraphs("руковод-\nства данными.")).toEqual(["руководства данными."]);
});

test("splitParagraphs: стопка закрывающих после терминатора — граница", () => {
  expect(splitParagraphs("Кончилось!»)\nНовое")).toEqual(["Кончилось!»)", "Новое"]);
});
