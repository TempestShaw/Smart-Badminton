import type { Metadata } from "next";
import "./globals.css";
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
    <html lang="zh-CN" className="font-sans antialiased" suppressHydrationWarning>
      <body>
        <ThemeProvider>
          <TooltipProvider>{children}</TooltipProvider>
          <Toaster richColors position="bottom-right" />
        </ThemeProvider>
      </body>
    </html>
  );
}
