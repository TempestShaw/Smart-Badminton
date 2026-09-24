"use client";

import { useEffect, useState, type ReactNode } from "react";
import { I18nextProvider, useTranslation } from "react-i18next";
import { Languages } from "lucide-react";
import { detectLocale, i18n, LANGUAGE_KEY } from "@/lib/i18n";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [ready, setReady] = useState(false);
  useEffect(() => {
    let saved: string | null = null;
    try { saved = localStorage.getItem(LANGUAGE_KEY); } catch { /* Storage may be disabled. */ }
    const updateDocument = (locale: string) => { document.documentElement.lang = locale === "zh" ? "zh-CN" : "en"; };
    i18n.on("languageChanged", updateDocument);
    void i18n.changeLanguage(detectLocale(saved, navigator.languages)).then(() => setReady(true));
    return () => { i18n.off("languageChanged", updateDocument); };
  }, []);
  return <I18nextProvider i18n={i18n}>{ready ? children : <div className="engine-loading" role="status" aria-label="Loading"><span className="loader" /></div>}</I18nextProvider>;
}

export function LanguageSwitcher() {
  const { t, i18n } = useTranslation();
  return (
    <Select value={i18n.resolvedLanguage ?? "en"} onValueChange={(locale) => {
      try { localStorage.setItem(LANGUAGE_KEY, locale); } catch { /* Switching still works without storage. */ }
      void i18n.changeLanguage(locale);
    }}>
      <SelectTrigger size="sm" className="language-switcher" aria-label={t("界面语言")}>
        <Languages aria-hidden="true" /><SelectValue />
      </SelectTrigger>
      <SelectContent position="popper" align="end">
        <SelectItem value="zh">中文</SelectItem>
        <SelectItem value="en">English</SelectItem>
      </SelectContent>
    </Select>
  );
}
