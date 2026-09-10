"use client";

import { useTranslation } from "react-i18next";
import { t } from "@/lib/i18n";

import { Moon, Sun } from "lucide-react";
import { ThemeProvider as NextThemeProvider, useTheme } from "next-themes";
import type { ReactNode } from "react";

import { Button } from "@/components/ui/button";

export function ThemeProvider({ children }: { children: ReactNode }) {
  return <NextThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange>{children}</NextThemeProvider>;
}

export function ThemeToggle() {
  useTranslation();
  const { resolvedTheme, setTheme } = useTheme();
  return (
    <Button className="theme-toggle" variant="outline" size="icon" aria-label={t("切换日间 / 夜间模式")} title={t("切换日间 / 夜间模式")} onClick={() => setTheme(resolvedTheme === "dark" ? "light" : "dark")}>
      <Sun className="theme-sun" /><Moon className="theme-moon" />
    </Button>
  );
}
