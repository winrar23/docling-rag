import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, basename, type SearchParams } from "@/api/client";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useDocuments } from "@/screens/documents/DocumentsScreen";

const selectCls =
  "border-input h-9 rounded-md border bg-transparent px-3 text-sm shadow-xs";

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
      q: HTMLInputElement; tag: HTMLSelectElement; topic: HTMLSelectElement; topk: HTMLInputElement;
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

  return (
    <div className="space-y-6 p-6">
      <h1 className="text-xl font-semibold">Поиск</h1>
      <form onSubmit={submit} className="flex flex-wrap items-end gap-3">
        <Input name="q" placeholder="Поисковый запрос…" className="min-w-64 flex-1" />
        <div className="space-y-1">
          <Label htmlFor="tag">Тег</Label>
          <select id="tag" name="tag" className={selectCls} defaultValue="">
            <option value="">Все</option>
            {allTags.map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
        </div>
        <div className="space-y-1">
          <Label htmlFor="topic">Тема</Label>
          <select id="topic" name="topic" className={selectCls} defaultValue="">
            <option value="">Все</option>
            {allTopics.map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
        </div>
        <div className="space-y-1">
          <Label htmlFor="topk">Top-k</Label>
          <Input id="topk" name="topk" type="number" min={1} max={50} placeholder="5" className="w-20" />
        </div>
        <Button type="submit">Найти</Button>
      </form>

      {search.isSuccess && search.data.results.length === 0 && (
        <p className="text-muted-foreground">Ничего не найдено.</p>
      )}
      <div className="space-y-3">
        {search.data?.results.map((r, i) => (
          <Card key={i}>
            <CardContent className="space-y-2 pt-4">
              <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
                <span className="font-medium text-foreground">{basename(r.source_file)}</span>
                {r.page_number != null && <span>стр. {r.page_number}</span>}
                {r.headings.length > 0 && <span>{r.headings.join(" → ")}</span>}
                {(r.element_type === "table" || r.element_type === "code") && (
                  <Badge variant="secondary">{r.element_type}</Badge>
                )}
                <span className="ml-auto tabular-nums">{r.score.toFixed(2)}</span>
              </div>
              <p className="whitespace-pre-wrap text-sm">{r.text}</p>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
