// Абзацы текста чанка: \n — граница только после конца предложения (плюс стопка
// закрывающих кавычек/скобок), остальные переносы — артефакты PDF-вёрстки:
// склеиваются пробелом, а перенос по дефису («руковод-» + «ства») — без пробела
// со стрипом дефиса.
const PARAGRAPH_END = /[.!?:;…]["»)\]]*$/;
const HYPHEN_BREAK = /\p{L}-$/u;

export function splitParagraphs(text: string): string[] {
  const paragraphs: string[] = [];
  let current = "";
  for (const line of text.split("\n")) {
    const fragment = line.trim();
    if (!fragment) continue;
    if (!current) {
      current = fragment;
    } else if (PARAGRAPH_END.test(current)) {
      paragraphs.push(current);
      current = fragment;
    } else if (HYPHEN_BREAK.test(current)) {
      current = current.slice(0, -1) + fragment;
    } else {
      current += " " + fragment;
    }
  }
  if (current) paragraphs.push(current);
  return paragraphs;
}
