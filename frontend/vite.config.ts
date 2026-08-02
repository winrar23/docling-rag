/// <reference types="vitest/config" />
import path from "node:path";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";

// Пути API, проксируемые на бэкенд в dev (docker api на :8000)
const backendPaths = ["/documents", "/jobs", "/search", "/chat", "/health"];

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(__dirname, "./src") } },
  server: {
    proxy: Object.fromEntries(backendPaths.map((p) => [p, "http://localhost:8000"])),
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "./src/test/setup.ts",
    css: false,
  },
});
