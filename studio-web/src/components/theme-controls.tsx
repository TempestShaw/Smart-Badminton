"use client";

import { Moon, Sun } from "lucide-react";
import { ThemeProvider as NextThemeProvider, useTheme } from "next-themes";
import type { ReactNode } from "react";

import { Button } from "@/components/ui/button";

export function ThemeProvider({ children }: { children: ReactNode }) {
  return <NextThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange>{children}</NextThemeProvider>;
}

export function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  return (
    <Button className="theme-toggle" variant="outline" size="icon" aria-label="切换日间 / 夜间模式" title="切换日间 / 夜间模式" onClick={() => setTheme(resolvedTheme === "dark" ? "light" : "dark")}>
      <Sun className="theme-sun" /><Moon className="theme-moon" />
    </Button>
  );
}
