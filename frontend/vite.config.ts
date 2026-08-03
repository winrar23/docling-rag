/// <reference types="vitest/config" />
import path from "node:path";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";

// Пути API, проксируемые на бэкенд в dev (docker api на :8000)
const backendPaths = ["/documents", "/jobs", "/search", "/chat", "/health"];

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(import.meta.dirname, "./src") } },
  server: {
    proxy: Object.fromEntries(backendPaths.map((p) => [p, "http://localhost:8000"])),
  },
  build: {
    rollupOptions: {
      output: {
        // Три вендорные группы (спека П.6). Пакет, не попавший в маски, уходит
        // в vendor — это безопасно: просто другой чанк, рантайм не меняется.
        manualChunks(id: string) {
          if (!id.includes("node_modules")) return undefined;
          if (/node_modules\/(react|react-dom|scheduler)\//.test(id)) return "react";
          if (
            /node_modules\/(react-markdown|remark-|rehype-|micromark|mdast-|hast-|unist-|unified|vfile|bail|trough|devlop|zwitch|ccount|longest-streak|markdown-table|character-entities|decode-named-character-reference|property-information|space-separated-tokens|comma-separated-tokens|html-url-attributes|style-to|inline-style-parser|estree-util|escape-string-regexp)/.test(id)
          ) {
            return "markdown";
          }
          return "vendor";
        },
      },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "./src/test/setup.ts",
    css: false,
  },
});
