import { useState } from "react";
import { FileText, MessageSquare, Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import ChatScreen from "@/screens/chat/ChatScreen";
import DocumentsScreen from "@/screens/documents/DocumentsScreen";
import SearchScreen from "@/screens/search/SearchScreen";

const SECTIONS = [
  { id: "chat", label: "Чат", icon: MessageSquare, screen: <ChatScreen /> },
  { id: "search", label: "Поиск", icon: Search, screen: <SearchScreen /> },
  { id: "documents", label: "Документы", icon: FileText, screen: <DocumentsScreen /> },
] as const;

type SectionId = (typeof SECTIONS)[number]["id"];

export default function App() {
  const [active, setActive] = useState<SectionId>("chat");
  return (
    <div className="flex h-screen">
      <aside className="flex w-56 shrink-0 flex-col gap-1 border-r p-3">
        <div className="mb-4 px-3 pt-2 font-semibold">docling-rag</div>
        {SECTIONS.map((s) => (
          <Button
            key={s.id}
            variant={s.id === active ? "secondary" : "ghost"}
            className="justify-start"
            onClick={() => setActive(s.id)}
          >
            <s.icon /> {s.label}
          </Button>
        ))}
      </aside>
      <main className="min-w-0 flex-1 overflow-y-auto">
        {SECTIONS.map((s) => (
          // все экраны смонтированы всегда: hidden сохраняет состояние чата/форм
          <div key={s.id} data-testid={`screen-${s.id}`} hidden={s.id !== active} className="h-full">
            {s.screen}
          </div>
        ))}
      </main>
    </div>
  );
}
