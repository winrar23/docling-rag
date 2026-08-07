import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { api } from "@/api/client";
import type { DocumentCard, DocumentPatchBody } from "@/api/types";
import { Button } from "@/components/ui/button";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export default function EditDialog({ card }: { card: DocumentCard }) {
  const [open, setOpen] = useState(false);
  const queryClient = useQueryClient();
  const patch = useMutation({
    mutationFn: (body: DocumentPatchBody) => api.patchDocument(card.id, body),
    onSuccess: () => {
      toast.success("Метаданные обновлены");
      queryClient.invalidateQueries({ queryKey: ["documents"] });
      setOpen(false);
    },
    // onError не нужен: глобальный MutationCache показывает toast (в т.ч. 409 при индексации)
  });

  const submit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const els = e.currentTarget.elements as typeof e.currentTarget.elements & {
      title: HTMLInputElement; author: HTMLInputElement;
      topic: HTMLInputElement; tags: HTMLInputElement;
    };
    patch.mutate({
      title: els.title.value.trim() || null,
      author: els.author.value.trim() || null,
      topic: els.topic.value.trim() || null,
      tags: els.tags.value.split(",").map((s) => s.trim()).filter(Boolean),
    });
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="ghost" size="sm">Редактировать</Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Метаданные документа</DialogTitle>
        </DialogHeader>
        <form onSubmit={submit} className="space-y-3">
          <div className="space-y-1">
            <Label htmlFor="title">Название</Label>
            <Input id="title" name="title" defaultValue={card.title ?? ""} />
          </div>
          <div className="space-y-1">
            <Label htmlFor="author">Автор</Label>
            <Input id="author" name="author" defaultValue={card.author ?? ""} />
          </div>
          <div className="space-y-1">
            <Label htmlFor="topic">Тема</Label>
            <Input id="topic" name="topic" defaultValue={card.topic ?? ""} />
          </div>
          <div className="space-y-1">
            <Label htmlFor="tags">Теги (через запятую)</Label>
            <Input id="tags" name="tags" defaultValue={card.tags.join(", ")} />
          </div>
          <Button type="submit" className="w-full" disabled={patch.isPending}>
            Сохранить
          </Button>
        </form>
      </DialogContent>
    </Dialog>
  );
}
