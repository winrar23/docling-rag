import { MutationCache, QueryCache, QueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { detailMessage } from "@/api/client";

// Единая точка конфигурации: боевой main.tsx и тестовый test/render.tsx создают
// клиент одинаково — тесты гоняют реальные дефолты, конфиг не расходится
export function makeQueryClient(): QueryClient {
  return new QueryClient({
    queryCache: new QueryCache({ onError: (e) => toast.error(detailMessage(e)) }),
    mutationCache: new MutationCache({ onError: (e) => toast.error(detailMessage(e)) }),
    defaultOptions: {
      queries: {
        retry: false, // 503 не долбить ретраями — там уже человекочитаемый detail
        // фокус окна не перезапрашивает: GET /search на каждый Alt-Tab писал бы дубли в searches
        refetchOnWindowFocus: false,
      },
    },
  });
}
