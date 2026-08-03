# docling-rag frontend

React SPA (этап 4-D): каталог документов, загрузка с прогрессом индексации,
семантический поиск, RAG-чат. В проде собирается frontend-стадией корневого
Dockerfile и раздаётся сервисом `api` с `http://localhost:8000/`.

Стек: Vite + React + TypeScript + Tailwind + shadcn/ui, TanStack Query;
тесты — Vitest + RTL + msw.

## Команды

```bash
npm install
npm test        # быстрые тесты
npm run dev     # vite :5173, прокси API на :8000 (нужен docker compose up api)
npm run build   # tsc -b + vite build → dist/
npm run lint    # oxlint
```

Архитектура и готчи — в корневом `CLAUDE.md` (раздел «Веб-UI (этап 4-D)»).
