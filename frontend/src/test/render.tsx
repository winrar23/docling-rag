import type { ReactElement } from "react";
import { MutationCache, QueryCache, QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import { toast } from "sonner";
import { Toaster } from "@/components/ui/sonner";
import { detailMessage } from "@/api/client";

export function renderWithClient(ui: ReactElement) {
  const client = new QueryClient({
    queryCache: new QueryCache({ onError: (e) => toast.error(detailMessage(e)) }),
    mutationCache: new MutationCache({ onError: (e) => toast.error(detailMessage(e)) }),
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      {ui}
      <Toaster />
    </QueryClientProvider>,
  );
}
