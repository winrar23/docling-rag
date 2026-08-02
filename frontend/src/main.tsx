import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { MutationCache, QueryCache, QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { toast } from "sonner";
import { Toaster } from "@/components/ui/sonner";
import { detailMessage } from "@/api/client";
import App from "./App";
import "./index.css";

const queryClient = new QueryClient({
  queryCache: new QueryCache({ onError: (e) => toast.error(detailMessage(e)) }),
  mutationCache: new MutationCache({ onError: (e) => toast.error(detailMessage(e)) }),
  defaultOptions: { queries: { retry: false } }, // 503 не долбить ретраями — там уже человекочитаемый detail
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
      <Toaster />
    </QueryClientProvider>
  </StrictMode>,
);
