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
      <aside className="flex w-56 shrink-0 flex-col gap-1 border-r bg-sidebar p-3">
        <div className="mb-4 flex items-center gap-2 px-3 pt-2 text-2xl font-semibold">
          <svg width="28" height="28" viewBox="0 0 40 40" aria-hidden="true"><rect x="6" y="8" width="7" height="24" rx="2" fill="var(--primary)"></rect><rect x="16" y="12" width="7" height="20" rx="2" fill="var(--primary)" opacity="0.65"></rect><rect x="26" y="5" width="7" height="27" rx="2" fill="var(--primary)" opacity="0.4"></rect><rect x="4" y="33" width="32" height="3" rx="1.5" fill="currentColor"></rect></svg>
          <span>
            Polka<span className="text-primary">.</span>
          </span>
        </div>
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
