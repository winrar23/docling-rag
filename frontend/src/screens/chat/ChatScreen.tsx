import { useEffect, useRef, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { ArrowUp, Eraser, X } from "lucide-react";
import ReactMarkdown from "react-markdown";
import { api } from "@/api/client";
import type { ChatSource, ChatTurn } from "@/api/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card";
import { Textarea } from "@/components/ui/textarea";
import { splitParagraphs } from "@/lib/paragraphs";

type Message = ChatTurn & { sources?: ChatSource[] };

function SourceChips({
  sources,
  onSelect,
}: {
  sources: ChatSource[];
  onSelect: (s: ChatSource) => void;
}) {
  if (sources.length === 0) return null;
  return (
    <div className="mt-3 flex flex-wrap gap-1.5">
      {sources.map((s, i) => (
        <HoverCard key={i} openDelay={0} closeDelay={0}>
          <HoverCardTrigger asChild>
            <Badge
              variant="outline"
              className="max-w-full cursor-pointer transition-colors hover:bg-muted"
              onClick={(e) => {
                e.stopPropagation(); // клик по чипу не должен считаться «кликом мимо» и закрывать панель
                onSelect(s);
              }}
            >
              <span className="truncate">📄 {s.file} · стр. {s.page}</span>
            </Badge>
          </HoverCardTrigger>
          <HoverCardContent className="w-80 text-sm">
            <p className="font-medium">{s.file}</p>
            {s.headings.length > 0 && (
              <p className="text-muted-foreground">{s.headings.join(" → ")}</p>
            )}
            <p className="text-muted-foreground">score {s.score.toFixed(2)}</p>
          </HoverCardContent>
        </HoverCard>
      ))}
    </div>
  );
}

export default function ChatScreen() {
  const [messages, setMessages] = useState<Message[]>([]);
  // Textarea — неконтролируемый (ref), не useState: React-DOM синхронизирует
  // element.defaultValue (= textContent) со `value` на КАЖДОМ рендере контролируемой
  // textarea без своего defaultValue-пропа (react-dom-client.development.js: updateTextarea());
  // из-за этого при откате текст, «стёртый» после отправки, каждый раз перерисовывался бы
  // обратно в дочерний текстовый узел DOM. Ref в обход этого читает/пишет .value напрямую.
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  // панель источника: source — содержимое (живёт во время анимации выезда), sourceOpen — видимость;
  // смена источника = выезд → подмена → въезд (200мс = duration-200)
  const [source, setSource] = useState<ChatSource | null>(null);
  const [sourceOpen, setSourceOpen] = useState(false);
  const switchTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const openSource = (s: ChatSource) => {
    if (switchTimer.current) clearTimeout(switchTimer.current);
    if (sourceOpen && s !== source) {
      setSourceOpen(false);
      switchTimer.current = setTimeout(() => {
        setSource(s);
        setSourceOpen(true);
      }, 200);
    } else {
      setSource(s);
      setSourceOpen(true);
    }
  };
  const closeSource = () => {
    if (switchTimer.current) clearTimeout(switchTimer.current);
    setSourceOpen(false);
  };

  // таймер подмены источника не должен стрелять после unmount
  useEffect(() => () => {
    if (switchTimer.current) clearTimeout(switchTimer.current);
  }, []);

  // Escape закрывает открытую панель; слушатель живёт только пока она открыта
  useEffect(() => {
    if (!sourceOpen) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (switchTimer.current) clearTimeout(switchTimer.current);
        setSourceOpen(false);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [sourceOpen]);

  const chat = useMutation({
    mutationFn: ({ message, history }: { message: string; history: ChatTurn[] }) =>
      api.chat(message, history),
    onSuccess: (res) =>
      setMessages((m) => [...m, { role: "assistant", content: res.answer, sources: res.sources }]),
    onError: (_e, vars) => {
      // toast покажет глобальный MutationCache; здесь — откат user-реплики и возврат текста
      setMessages((m) => m.slice(0, -1));
      if (textareaRef.current) textareaRef.current.value = vars.message;
    },
  });

  const send = () => {
    const el = textareaRef.current;
    const text = el?.value.trim() ?? "";
    if (!text || chat.isPending) return;
    const history: ChatTurn[] = messages.map(({ role, content }) => ({ role, content }));
    setMessages((m) => [...m, { role: "user", content: text }]);
    if (el) el.value = "";
    chat.mutate({ message: text, history });
  };

  return (
    <div className="mx-auto flex h-full max-w-3xl flex-col p-6" onClick={closeSource}>
      <div className="flex items-center justify-between border-b pb-3">
        <h1 className="text-xl font-semibold">Чат</h1>
        <Button variant="ghost" size="sm" onClick={() => setMessages([])} disabled={chat.isPending}>
          <Eraser /> Очистить диалог
        </Button>
      </div>
      <div className="flex-1 space-y-4 overflow-y-auto py-4">
        {messages.length === 0 && (
          <p className="pt-24 text-center text-sm text-muted-foreground">Задай вопрос по проиндексированной документации.</p>
        )}
        {messages.map((m, i) =>
          m.role === "user" ? (
            <div key={i} className="ml-auto w-fit max-w-[80%] rounded-2xl rounded-br-md bg-secondary px-4 py-2.5">
              {m.content}
            </div>
          ) : (
            <div key={i} className="max-w-[90%] rounded-2xl rounded-bl-md border bg-card px-4 py-3 shadow-xs">
              <div className="space-y-2 text-sm leading-relaxed">
                <ReactMarkdown>{m.content}</ReactMarkdown>
              </div>
              <SourceChips sources={m.sources ?? []} onSelect={openSource} />
            </div>
          ),
        )}
        {chat.isPending && <p className="text-muted-foreground">Думает…</p>}
      </div>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          send();
        }}
        className="flex items-center gap-2 rounded-full border border-input p-1.5 pl-4 transition-colors focus-within:border-ring focus-within:ring-3 focus-within:ring-ring/50"
      >
        <Textarea
          ref={textareaRef}
          defaultValue=""
          placeholder="Спросить по документации…"
          rows={1}
          className="min-h-0 resize-none border-0 bg-transparent p-0 shadow-none focus-visible:border-transparent focus-visible:ring-0 dark:bg-transparent"
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
        />
        <Button
          type="submit"
          variant="secondary"
          aria-label="Отправить"
          className="size-8 shrink-0 rounded-full"
          disabled={chat.isPending}
        >
          <ArrowUp />
        </Button>
      </form>
      <aside
        aria-label="Источник"
        data-testid="source-panel"
        inert={!sourceOpen}
        onClick={(e) => e.stopPropagation()}
        className={`fixed inset-y-0 right-0 z-40 flex w-[48rem] max-w-full flex-col border-l bg-background shadow-lg transition-transform duration-200 ${
          sourceOpen ? "translate-x-0" : "translate-x-full"
        }`}
      >
        <div className="flex items-center justify-between gap-2 border-b p-4">
          <p className="min-w-0 truncate font-medium">📄 {source?.file}</p>
          <Button variant="ghost" size="sm" aria-label="Закрыть" onClick={closeSource}>
            <X />
          </Button>
        </div>
        {source && (
          <div className="flex-1 space-y-4 overflow-y-auto p-4 text-sm">
            {source.headings.length > 0 && (
              <p className="text-muted-foreground">{source.headings.join(" → ")}</p>
            )}
            <div className="flex gap-4 text-muted-foreground">
              <span>стр. {source.page}</span>
              <span>score {source.score.toFixed(2)}</span>
            </div>
            {source.text ? (
              source.element_type === "table" || source.element_type === "code" ? (
                <p className="leading-relaxed whitespace-pre-wrap">{source.text}</p>
              ) : (
                <div className="space-y-3">
                  {splitParagraphs(source.text).map((para, i) => (
                    <p key={i} className="leading-relaxed">
                      {para}
                    </p>
                  ))}
                </div>
              )
            ) : (
              <p className="text-muted-foreground">
                Текст фрагмента недоступен — бэкенд не вернул поле text.
              </p>
            )}
          </div>
        )}
      </aside>
    </div>
  );
}
