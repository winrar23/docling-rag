import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { FileText, Search } from "lucide-react";
import { api, basename, type SearchParams } from "@/api/client";
import type { SearchResult } from "@/api/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useDocuments } from "@/api/hooks";
import { plural } from "@/lib/plural";

const chipCls =
  "border-input text-muted-foreground flex h-8 items-center gap-1.5 rounded-full border bg-background px-3 text-[13px] whitespace-nowrap";
const chipSelectCls =
  "text-foreground bg-transparent font-medium outline-none";

export default function SearchScreen() {
  const [params, setParams] = useState<SearchParams | null>(null);
  const documents = useDocuments();
  const cards = documents.data ?? [];
  const allTags = [...new Set(cards.flatMap((c) => c.tags))].sort();
  const allTopics = [...new Set(cards.map((c) => c.topic).filter((t): t is string => !!t))].sort();

  const search = useQuery({
    queryKey: ["search", params],
    queryFn: () => api.search(params!),
    enabled: params !== null,
  });

  const submit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const els = e.currentTarget.elements as typeof e.currentTarget.elements & {
      q: HTMLInputElement; tag: HTMLSelectElement; topic: HTMLSelectElement; topk: HTMLSelectElement;
    };
    const q = els.q.value.trim();
    if (!q) return;
    const next: SearchParams = {
      q,
      tag: els.tag.value || undefined,
      topic: els.topic.value || undefined,
      topK: els.topk.value ? Number(els.topk.value) : undefined,
    };
    // идентичные параметры дают тот же queryKey — TanStack молча отдал бы кеш;
    // повторный сабмит должен перезапросить (и записать поиск в searches), как повторный cli search
    const same =
      params !== null &&
      next.q === params.q &&
      next.tag === params.tag &&
      next.topic === params.topic &&
      next.topK === params.topK;
    if (same) void search.refetch();
    else setParams(next);
  };

  const results = search.data?.results ?? [];
  // Группировка по документу с сохранением порядка релевантности
  const groups: { file: string; items: SearchResult[] }[] = [];
  for (const r of results) {
    const g = groups.find((g) => g.file === r.source_file);
    if (g) g.items.push(r);
    else groups.push({ file: r.source_file, items: [r] });
  }

  return (
    <div className="mx-auto max-w-3xl space-y-5 p-6 pt-10">
      <form onSubmit={submit} className="space-y-3">
        <div className="flex gap-2.5">
          <div className="relative flex-1">
            <Search className="text-muted-foreground pointer-events-none absolute top-1/2 left-3.5 size-4 -translate-y-1/2" />
            <Input name="q" placeholder="Поисковый запрос…" className="h-9.5 rounded-lg pl-10" />
          </div>
          <Button type="submit" className="h-9.5 rounded-lg px-5">Найти</Button>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <label className={chipCls}>
            Тег
            <select name="tag" defaultValue="" className={chipSelectCls}>
              <option value="">Все</option>
              {allTags.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
          </label>
          <label className={chipCls}>
            Тема
            <select name="topic" defaultValue="" className={chipSelectCls}>
              <option value="">Все</option>
              {allTopics.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
          </label>
          <label className={chipCls}>
            Top-k
            <select name="topk" defaultValue="" className={chipSelectCls}>
              <option value="">5</option>
              {[3, 10, 20, 50].map((n) => <option key={n} value={n}>{n}</option>)}
            </select>
          </label>
          {search.isSuccess && (
            <span className="text-muted-foreground ml-auto text-[13px] tabular-nums">
              {results.length} {plural(results.length, "фрагмент", "фрагмента", "фрагментов")}
            </span>
          )}
        </div>
      </form>

      {search.isSuccess && results.length === 0 && (
        <p className="text-muted-foreground pt-4">Ничего не найдено.</p>
      )}
      <div className="space-y-6 pt-2">
        {groups.map((g) => (
          <section key={g.file} className="space-y-2.5">
            <div className="flex items-center gap-2 text-sm font-medium">
              <FileText className="text-muted-foreground size-3.5" />
              {basename(g.file)}
              <Badge variant="secondary" className="rounded-full font-normal">
                {g.items.length} {plural(g.items.length, "фрагмент", "фрагмента", "фрагментов")}
              </Badge>
            </div>
            <div className="space-y-2.5">
              {g.items.map((r, i) => (
                <div key={i} className="rounded-xl border px-4 pt-3.5 pb-4">
                  <div className="text-muted-foreground flex flex-wrap items-center gap-2 text-[13px]">
                    {r.page_number != null && <span>стр. {r.page_number}</span>}
                    {r.headings.length > 0 && <span>· {r.headings.join(" → ")}</span>}
                    {(r.element_type === "table" || r.element_type === "code") && (
                      <Badge variant="secondary" className="rounded-full font-normal">{r.element_type}</Badge>
                    )}
                    <span className="ml-auto flex items-center gap-2">
                      <span className="bg-muted inline-block h-1 w-16 overflow-hidden rounded-full">
                        <span
                          className="bg-primary block h-full"
                          style={{ width: `${Math.round(Math.min(Math.max(r.score, 0), 1) * 100)}%` }}
                        />
                      </span>
                      <span className="text-foreground font-medium tabular-nums">{r.score.toFixed(2)}</span>
                    </span>
                  </div>
                  <p className="mt-2 text-sm leading-relaxed whitespace-pre-wrap">{r.text}</p>
                </div>
              ))}
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}
