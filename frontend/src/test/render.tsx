import type { ReactElement } from "react";
import { QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import { Toaster } from "@/components/ui/sonner";
import { makeQueryClient } from "@/lib/queryClient";

export function renderWithClient(ui: ReactElement) {
  return render(
    <QueryClientProvider client={makeQueryClient()}>
      {ui}
      <Toaster />
    </QueryClientProvider>,
  );
}
