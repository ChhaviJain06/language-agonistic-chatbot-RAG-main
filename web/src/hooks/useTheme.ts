import { useCallback, useEffect, useState } from "react";

export type Theme = "light" | "dark";
const KEY = "bhasha-theme";

function systemTheme(): Theme {
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function stored(): Theme | null {
  try {
    const v = localStorage.getItem(KEY);
    return v === "light" || v === "dark" ? v : null;
  } catch {
    return null;
  }
}

export function useTheme() {
  const [theme, setTheme] = useState<Theme>(() => stored() ?? systemTheme());
  const [manual, setManual] = useState<boolean>(() => stored() !== null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  useEffect(() => {
    if (manual) return;
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const on = () => setTheme(mq.matches ? "dark" : "light");
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, [manual]);

  const toggle = useCallback(() => {
    const next: Theme = theme === "dark" ? "light" : "dark";
    setTheme(next);
    setManual(true);
    try {
      localStorage.setItem(KEY, next);
    } catch {
      /* storage unavailable */
    }
  }, [theme]);

  return { theme, toggle };
}
