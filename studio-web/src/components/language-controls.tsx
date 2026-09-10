"use client";

import { useEffect, useState, type ReactNode } from "react";
import { I18nextProvider, useTranslation } from "react-i18next";
import { Languages } from "lucide-react";
import { detectLocale, i18n, LANGUAGE_KEY } from "@/lib/i18n";

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
    <label className="language-switcher">
      <Languages aria-hidden="true" />
      <select aria-label={t("界面语言")} value={i18n.resolvedLanguage ?? "en"} onChange={(event) => {
        const locale = event.target.value;
        try { localStorage.setItem(LANGUAGE_KEY, locale); } catch { /* Switching still works without storage. */ }
        void i18n.changeLanguage(locale);
      }}>
        <option value="zh">中文</option>
        <option value="en">English</option>
      </select>
    </label>
  );
}
