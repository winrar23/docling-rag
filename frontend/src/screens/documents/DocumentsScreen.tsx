import { useEffect, useRef } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { api, basename } from "@/api/client";
import type { DocumentCard, Job } from "@/api/types";
import { isActive, useDocuments, useJobs } from "@/api/hooks";
import {
  AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent,
  AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import EditDialog from "@/screens/documents/EditDialog";
import UploadDialog from "@/screens/documents/UploadDialog";

// step из БД null до первого прогресс-тика — фолбэк обязан быть русским (Global Constraints)
const STATUS_RU: Record<Job["status"], string> = {
  queued: "в очереди",
  running: "выполняется",
  done: "готово",
  failed: "ошибка",
};

function IndexingSection({ jobs }: { jobs: Job[] }) {
  const visible = jobs.filter((j) => j.status !== "done");
  if (visible.length === 0) return null;
  return (
    <section className="space-y-3">
      <h2 className="font-medium">Индексация</h2>
      {visible.map((j) => (
        <div key={j.id} className="rounded-md border p-3">
          <div className="flex items-baseline justify-between">
            <span className="font-medium">{j.original_name}</span>
            <span className="text-sm text-muted-foreground">
              {j.status === "failed" ? "ошибка" : j.step ?? STATUS_RU[j.status]}
            </span>
          </div>
          {j.status === "failed" ? (
            <p className="mt-1 text-sm text-destructive">{j.error}</p>
          ) : (
            <div className="mt-2 flex items-center gap-3">
              <Progress
                value={j.chunks_total ? ((j.chunks_done ?? 0) / j.chunks_total) * 100 : 0}
                className="h-2"
              />
              <span className="whitespace-nowrap text-sm text-muted-foreground">
                {j.chunks_done ?? 0} / {j.chunks_total ?? "?"}
              </span>
            </div>
          )}
        </div>
      ))}
    </section>
  );
}

function DeleteButton({ card }: { card: DocumentCard }) {
  const queryClient = useQueryClient();
  const del = useMutation({
    mutationFn: () => api.deleteDocument(card.id),
    onSuccess: (res) => {
      toast.success(`Удалено: ${res.deleted} (${res.chunks} chunks)`);
      queryClient.invalidateQueries({ queryKey: ["documents"] });
    },
    // onError не нужен: 409/прочие ошибки показывает глобальный MutationCache toast
  });
  const name = card.title ?? basename(card.source_file);
  return (
    <AlertDialog>
      <AlertDialogTrigger asChild>
        <Button variant="ghost" size="sm">Удалить</Button>
      </AlertDialogTrigger>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Удалить «{name}»?</AlertDialogTitle>
          <AlertDialogDescription>
            Запись, чанки и файл из uploads будут удалены. История джоб и лог поисков останутся.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Отмена</AlertDialogCancel>
          <AlertDialogAction onClick={() => del.mutate()}>Да, удалить</AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

export default function DocumentsScreen() {
  const queryClient = useQueryClient();
  const documents = useDocuments();
  const jobs = useJobs();

  // активная джоба завершилась → каталог мог пополниться
  const prevActive = useRef<Set<string>>(new Set());
  useEffect(() => {
    const list = jobs.data ?? [];
    const finished = list.some((j) => prevActive.current.has(j.id) && !isActive(j));
    if (finished) queryClient.invalidateQueries({ queryKey: ["documents"] });
    prevActive.current = new Set(list.filter(isActive).map((j) => j.id));
  }, [jobs.data, queryClient]);

  const cards = documents.data ?? [];
  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">Документы</h1>
        <UploadDialog />
      </div>
      <IndexingSection jobs={jobs.data ?? []} />
      {documents.isSuccess && cards.length === 0 ? (
        <p className="text-muted-foreground">Хранилище пустое. Документов нет.</p>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Название</TableHead>
              <TableHead>Автор</TableHead>
              <TableHead>Тема</TableHead>
              <TableHead>Теги</TableHead>
              <TableHead>Добавлен</TableHead>
              <TableHead className="text-right">Чанков</TableHead>
              <TableHead className="text-right">Действия</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {cards.map((c) => (
              <TableRow key={c.id}>
                <TableCell title={c.source_file}>{c.title ?? basename(c.source_file)}</TableCell>
                <TableCell>{c.author ?? "—"}</TableCell>
                <TableCell>{c.topic ?? "—"}</TableCell>
                <TableCell className="space-x-1">
                  {c.tags.map((t) => (
                    <Badge key={t} variant="outline">{t}</Badge>
                  ))}
                </TableCell>
                <TableCell>{new Date(c.added_at).toLocaleString("ru-RU")}</TableCell>
                <TableCell className="text-right">{c.chunks}</TableCell>
                <TableCell className="text-right">
                  <EditDialog card={c} />
                  <DeleteButton card={c} />
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
    </div>
  );
}
