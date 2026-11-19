import { useCallback, useEffect, useState } from "react";
import { fetchHealth } from "../lib/api";
import type { Health } from "../lib/types";

export function useHealth(intervalMs = 15000) {
  const [health, setHealth] = useState<Health | null>(null);
  const [down, setDown] = useState(false);
  const [checking, setChecking] = useState(true);

  const check = useCallback(async (signal?: AbortSignal) => {
    setChecking(true);
    try {
      const h = await fetchHealth(signal);
      setHealth(h);
      setDown(false);
    } catch {
      if (!signal?.aborted) setDown(true);
    } finally {
      if (!signal?.aborted) setChecking(false);
    }
  }, []);

  useEffect(() => {
    const ac = new AbortController();
    void check(ac.signal);
    const t = window.setInterval(() => void check(ac.signal), intervalMs);
    return () => {
      ac.abort();
      window.clearInterval(t);
    };
  }, [check, intervalMs]);

  return { health, down, checking, retry: () => void check() };
}
