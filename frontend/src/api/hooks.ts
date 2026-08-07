import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import type { Job } from "@/api/types";

export function useDocuments() {
  return useQuery({ queryKey: ["documents"], queryFn: api.listDocuments });
}

export const isActive = (j: Job) => j.status === "queued" || j.status === "running";

export function useJobs() {
  return useQuery({
    queryKey: ["jobs"],
    queryFn: () => api.listJobs(),
    // поллинг только пока есть активные джобы; иначе не дёргаем API
    refetchInterval: (query) => (query.state.data?.some(isActive) ? 2000 : false),
  });
}
