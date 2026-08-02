import { useQuery } from "@tanstack/react-query";
import { api, basename } from "@/api/client";
import { Badge } from "@/components/ui/badge";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";

export function useDocuments() {
  return useQuery({ queryKey: ["documents"], queryFn: api.listDocuments });
}

export function useJobs() {
  return useQuery({ queryKey: ["jobs"], queryFn: () => api.listJobs() });
}

export default function DocumentsScreen() {
  const documents = useDocuments();
  const jobs = useJobs(); // прогрев для секции «Индексация» (Task 6)
  void jobs;
  const cards = documents.data ?? [];
  return (
    <div className="space-y-6 p-6">
      <h1 className="text-xl font-semibold">Документы</h1>
      {documents.isSuccess && cards.length === 0 ? (
        <p className="text-muted-foreground">Хранилище пустое. Документов нет.</p>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Название</TableHead>
              <TableHead>Тема</TableHead>
              <TableHead>Теги</TableHead>
              <TableHead>Добавлен</TableHead>
              <TableHead className="text-right">Чанков</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {cards.map((c) => (
              <TableRow key={c.id}>
                <TableCell title={c.source_file}>{c.title ?? basename(c.source_file)}</TableCell>
                <TableCell>{c.topic}</TableCell>
                <TableCell className="space-x-1">
                  {c.tags.map((t) => (
                    <Badge key={t} variant="outline">{t}</Badge>
                  ))}
                </TableCell>
                <TableCell>{new Date(c.added_at).toLocaleString("ru-RU")}</TableCell>
                <TableCell className="text-right">{c.chunks}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
    </div>
  );
}
