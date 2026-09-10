import type { Metadata } from "next";
import "./globals.css";
import { LanguageProvider } from "@/components/language-controls";
import { ThemeProvider } from "@/components/theme-controls";
import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";

export const metadata: Metadata = {
  title: "Smart Badminton Studio",
  description: "Local-first badminton rally editor",
  icons: { icon: process.env.STUDIO_EMBEDDED_BUILD === "1" ? "/static/icon.svg" : "/icon.svg" },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="font-sans antialiased" suppressHydrationWarning>
      <body>
        <LanguageProvider>
          <ThemeProvider>
            <TooltipProvider>{children}</TooltipProvider>
            <Toaster richColors position="bottom-right" />
          </ThemeProvider>
        </LanguageProvider>
      </body>
    </html>
  );
}
