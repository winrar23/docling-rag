import { useRef, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import ReactMarkdown from "react-markdown";
import { api } from "@/api/client";
import type { ChatSource, ChatTurn } from "@/api/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card";
import { Textarea } from "@/components/ui/textarea";

type Message = ChatTurn & { sources?: ChatSource[] };

function SourceChips({ sources }: { sources: ChatSource[] }) {
  if (sources.length === 0) return null;
  return (
    <div className="mt-2 flex flex-wrap gap-1.5">
      {sources.map((s, i) => (
        <HoverCard key={i} openDelay={0} closeDelay={0}>
          <HoverCardTrigger asChild>
            <Badge variant="outline" className="cursor-default">
              📄 {s.file} · стр. {s.page}
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
    <div className="mx-auto flex h-full max-w-3xl flex-col p-6">
      <div className="mb-3 flex items-center justify-between">
        <h1 className="text-xl font-semibold">Чат</h1>
        <Button variant="ghost" size="sm" onClick={() => setMessages([])}>
          Новый диалог
        </Button>
      </div>
      <div className="flex-1 space-y-4 overflow-y-auto pb-4">
        {messages.length === 0 && (
          <p className="text-muted-foreground">Задай вопрос по проиндексированной документации.</p>
        )}
        {messages.map((m, i) =>
          m.role === "user" ? (
            <div key={i} className="ml-auto w-fit max-w-[80%] rounded-lg bg-secondary px-3 py-2">
              {m.content}
            </div>
          ) : (
            <div key={i} className="max-w-[90%] rounded-lg border px-3 py-2">
              <div className="text-sm">
                <ReactMarkdown>{m.content}</ReactMarkdown>
              </div>
              <SourceChips sources={m.sources ?? []} />
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
        className="flex gap-2"
      >
        <Textarea
          ref={textareaRef}
          defaultValue=""
          placeholder="Спросить по документации…"
          rows={2}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
        />
        <Button type="submit" disabled={chat.isPending}>
          Отправить
        </Button>
      </form>
    </div>
  );
}
