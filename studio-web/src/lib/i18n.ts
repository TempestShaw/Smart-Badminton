import i18next, { type TOptions } from "i18next";
import { initReactI18next } from "react-i18next";
import en from "../locales/en.json";
import zh from "../locales/zh.json";

export type Locale = "en" | "zh";
export const LANGUAGE_KEY = "smart-badminton:language";

export function detectLocale(saved: string | null, languages: readonly string[]): Locale {
  if (saved === "en" || saved === "zh") return saved;
  for (const language of languages) {
    const base = language.toLowerCase().split("-")[0];
    if (base === "zh" || base === "en") return base;
  }
  return "en";
}

export const i18n = i18next.createInstance();
void i18n.use(initReactI18next).init({
  lng: "en",
  fallbackLng: "en",
  supportedLngs: ["en", "zh"],
  resources: { en: { translation: en }, zh: { translation: zh } },
  keySeparator: false,
  nsSeparator: false,
  interpolation: { escapeValue: false },
  initAsync: false,
});

// The stable translator also serves asynchronous callbacks; React views subscribe via useTranslation.
export const t = (key: string, options?: TOptions): string => String(i18n.t(key, options));

// Local Studio returns human-readable messages. Translate known messages at the
// display boundary, preserving unknown diagnostics and user-supplied filenames.
const sourceKeys = new Map(Object.entries(en).map(([key, value]) => [value, key]));
const escapeRegex = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const messagePatterns = Object.entries(en).flatMap(([key, english]) => {
  if (!key.includes("{{")) return [];
  return [key, english].map((template) => {
    const names: string[] = [];
    const parts = template.split(/(\{\{\w+\}\})/g);
    const pattern = parts.map((part) => {
      if (!part.startsWith("{{")) return escapeRegex(part);
      names.push(part.slice(2, -2));
      return "(.+?)";
    }).join("");
    return { key, names, regex: new RegExp(`^${pattern}$`) };
  });
});

export function translateMessage(message: string | undefined | null): string {
  if (!message) return "";
  if (Object.hasOwn(en, message)) return t(message);
  const source = sourceKeys.get(message);
  if (source) return t(source);
  for (const { key, names, regex } of messagePatterns) {
    const match = regex.exec(message);
    if (match) return t(key, Object.fromEntries(names.map((name, index) => [name, translateMessage(match[index + 1])])));
  }
  return message;
}
