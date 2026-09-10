import { describe, expect, it } from "vitest";
import { detectLocale, i18n, t, translateMessage } from "./i18n";
import en from "../locales/en.json";
import zh from "../locales/zh.json";

// Both static builds share this catalog, browser detection and runtime messages.
describe("Studio language selection", () => {
  it("honors explicit preference, Chinese variants and browser priority", () => {
    expect(detectLocale("zh", ["en-US"])).toBe("zh");
    expect(detectLocale("en", ["zh-TW"])).toBe("en");
    expect(detectLocale(null, ["zh-Hant-HK", "en"])).toBe("zh");
    expect(detectLocale(null, ["en-GB", "zh"])).toBe("en");
    expect(detectLocale("invalid", ["fr-FR", "zh-CN"])).toBe("zh");
    expect(detectLocale(null, ["fr-FR"])).toBe("en");
  });

  it("keeps complete catalogs with matching interpolation placeholders", () => {
    expect(Object.keys(en).sort()).toEqual(Object.keys(zh).sort());
    for (const [key, english] of Object.entries(en)) {
      expect(english.trim(), key).not.toBe("");
      const fields = (value: string) => value.match(/\{\{\w+\}\}/g)?.sort() ?? [];
      expect(fields(english), key).toEqual(fields(zh[key as keyof typeof zh]));
    }
  });

  it("translates both engines, cached progress and native labels on language change", async () => {
    await i18n.changeLanguage("en");
    expect(t("新建剪片")).toBe("New project");
    expect(t("保存校准")).toBe("Save calibration");
    expect(translateMessage("分析画面 42%")).toBe("Analyzing frames: 42%");
    expect(translateMessage("正在运行 Hybrid 羽球检测")).toBe("Running Hybrid shuttle detection");
    expect(translateMessage("背景排除区 2")).toBe("Background exclusion 2");
    expect(translateMessage("Practice-court.mp4")).toBe("Practice-court.mp4");
    await i18n.changeLanguage("zh");
    expect(translateMessage("Analyzing frames: 42%")).toBe("分析画面 42%");
    expect(translateMessage("Background exclusion 2")).toBe("背景排除区 2");
    expect(t("保存校准")).toBe("保存校准");
  });
});
