import { useRef, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { api } from "@/api/client";
import { Button } from "@/components/ui/button";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

// нативный select: Radix Select нестабилен в jsdom (hasPointerCapture) — осознанный выбор
const selectCls =
  "border-input h-9 w-full rounded-md border bg-transparent px-3 text-sm shadow-xs";

export default function UploadDialog() {
  const [open, setOpen] = useState(false);
  const formRef = useRef<HTMLFormElement>(null);
  const queryClient = useQueryClient();
  const upload = useMutation({
    mutationFn: api.upload,
    onSuccess: () => {
      toast.success("Файл в очереди индексации");
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
      setOpen(false);
    },
    // onError не нужен: глобальный MutationCache уже показывает toast с detail
  });

  const submit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const fd = new FormData();
    const els = e.currentTarget.elements as typeof e.currentTarget.elements & {
      file: HTMLInputElement; title: HTMLInputElement; topic: HTMLInputElement;
      tags: HTMLInputElement; ocr: HTMLSelectElement; ocr_lang: HTMLSelectElement;
    };
    const file = els.file.files?.[0];
    if (!file) {
      // noValidate убрал нативную браузерную валидацию (нужно для jsdom, см. комментарий на <form>) —
      // без явного toast сабмит без файла был бы молчаливым no-op
      toast.error("Выберите файл");
      return;
    }
    fd.append("file", file);
    if (els.title.value.trim()) fd.append("title", els.title.value.trim());
    if (els.topic.value.trim()) fd.append("topic", els.topic.value.trim());
    for (const t of els.tags.value.split(",").map((s) => s.trim()).filter(Boolean)) {
      fd.append("tags", t);
    }
    fd.append("ocr", els.ocr.value);
    fd.append("ocr_lang", els.ocr_lang.value);
    upload.mutate(fd);
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button>Загрузить документ</Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Загрузка документа</DialogTitle>
        </DialogHeader>
        {/* noValidate: jsdom считает required-file-input невалидным даже после userEvent.upload
            (checkValidity() всегда false для type=file), из-за чего click по submit не диспатчит
            submit-событие в тестах — required остаётся для семантики/aria, проверка файла ручная (see below) */}
        <form ref={formRef} onSubmit={submit} noValidate className="space-y-3">
          <div className="space-y-1">
            <Label htmlFor="file">Файл (PDF, DOCX, MD)</Label>
            <Input id="file" name="file" type="file" accept=".pdf,.docx,.md" required />
          </div>
          <div className="space-y-1">
            <Label htmlFor="title">Название</Label>
            <Input id="title" name="title" placeholder="Опционально" />
          </div>
          <div className="space-y-1">
            <Label htmlFor="topic">Тема</Label>
            <Input id="topic" name="topic" placeholder="Опционально" />
          </div>
          <div className="space-y-1">
            <Label htmlFor="tags">Теги (через запятую)</Label>
            <Input id="tags" name="tags" placeholder="arch, solid" />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <Label htmlFor="ocr">Режим OCR</Label>
              <select id="ocr" name="ocr" defaultValue="auto" className={selectCls}>
                <option value="auto">Авто (по текстовому слою)</option>
                <option value="on">Принудительно</option>
                <option value="off">Выключен</option>
              </select>
            </div>
            <div className="space-y-1">
              <Label htmlFor="ocr_lang">Язык OCR</Label>
              <select id="ocr_lang" name="ocr_lang" defaultValue="en" className={selectCls}>
                <option value="en">Английский</option>
                <option value="ru">Русский</option>
              </select>
            </div>
          </div>
          <Button type="submit" className="w-full" disabled={upload.isPending}>
            Загрузить
          </Button>
        </form>
      </DialogContent>
    </Dialog>
  );
}
