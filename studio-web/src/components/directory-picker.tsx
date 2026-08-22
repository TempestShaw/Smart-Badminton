"use client";

import { useCallback, useEffect, useState } from "react";
import { ChevronLeft, Folder, FolderOpen, HardDrive, Home, LoaderCircle, RefreshCw } from "lucide-react";

import { apiRequest } from "@/lib/api";
import type { DirectoryPayload } from "@/types/studio";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";

export function DirectoryPicker({
  open,
  onOpenChange,
  initialPath,
  title,
  description,
  onSelect,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  initialPath: string;
  title: string;
  description: string;
  onSelect: (path: string) => boolean | void | Promise<boolean | void>;
}) {
  const [browser, setBrowser] = useState<DirectoryPayload | null>(null);
  const [path, setPath] = useState(initialPath);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const load = useCallback(async (target?: string) => {
    try {
      setLoading(true);
      setError("");
      const query = target?.trim() ? `?path=${encodeURIComponent(target.trim())}` : "";
      const payload = await apiRequest<DirectoryPayload>(`/api/filesystem/directories${query}`);
      setBrowser(payload);
      setPath(payload.path);
    } catch (caught) {
      setError((caught as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!open) return;
    setPath(initialPath);
    void load(initialPath);
  }, [initialPath, load, open]);

  const choose = async () => {
    if (!browser) return;
    const accepted = await onSelect(browser.path);
    if (accepted !== false) onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl border-border bg-popover text-popover-foreground">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2"><FolderOpen className="size-5 text-primary" />{title}</DialogTitle>
          <DialogDescription>{description}</DialogDescription>
        </DialogHeader>

        <div className="flex gap-2">
          <Input
            aria-label="文件夹路径"
            value={path}
            spellCheck={false}
            onChange={(event) => setPath(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") void load(path);
            }}
          />
          <Button variant="outline" disabled={loading} onClick={() => void load(path)}>
            {loading ? <LoaderCircle className="animate-spin" /> : <RefreshCw />}
            打开
          </Button>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <Button size="sm" variant="outline" disabled={!browser?.parent || loading} onClick={() => void load(browser?.parent ?? undefined)}>
            <ChevronLeft />上一级
          </Button>
          <Button size="sm" variant="outline" disabled={!browser?.home || loading} onClick={() => void load(browser?.home)}>
            <Home />主目录
          </Button>
          {(browser?.roots ?? []).map((root) => (
            <Button key={root.path} size="sm" variant="ghost" disabled={loading} onClick={() => void load(root.path)}>
              <HardDrive />{root.name}
            </Button>
          ))}
          {browser ? <Badge variant="secondary" className="ml-auto max-w-full truncate font-mono">{browser.path}</Badge> : null}
        </div>

        {error ? <div role="alert" className="rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">{error}</div> : null}

        <ScrollArea className="h-80 rounded-lg border bg-background/40">
          <div className="grid gap-1 p-2">
            {!loading && browser?.directories.length === 0 ? (
              <p className="p-6 text-center text-sm text-muted-foreground">这个文件夹没有子文件夹；仍然可以选择当前文件夹。</p>
            ) : null}
            {(browser?.directories ?? []).map((directory) => (
              <Button
                key={directory.path}
                variant="ghost"
                className="h-10 justify-start gap-3 px-3 font-normal"
                onClick={() => void load(directory.path)}
              >
                <Folder className="size-4 text-primary" />
                <span className="truncate">{directory.name}</span>
              </Button>
            ))}
          </div>
        </ScrollArea>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>取消</Button>
          <Button disabled={!browser || loading} onClick={() => void choose()}>
            <FolderOpen />选择当前文件夹
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
