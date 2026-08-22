"use client";

import { useEffect, useState } from "react";
import { BookOpenText } from "lucide-react";
import Markdown from "react-markdown";

import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { apiRequest } from "@/lib/api";

export function StudioTutorial() {
  const [open, setOpen] = useState(false);
  const [markdown, setMarkdown] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    if (!open || markdown || error) return;
    let active = true;
    apiRequest<{ markdown: string }>("/api/tutorial")
      .then((payload) => {
        if (active) setMarkdown(payload.markdown);
      })
      .catch((reason: Error) => {
        if (active) setError(reason.message);
      });
    return () => {
      active = false;
    };
  }, [error, markdown, open]);

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button size="sm" variant="ghost" className="tutorial-tab"><BookOpenText />使用教程</Button>
      </DialogTrigger>
      <DialogContent className="tutorial-dialog max-w-5xl">
        <DialogHeader>
          <DialogTitle>Smart Badminton Studio 使用教程</DialogTitle>
          <DialogDescription>从选择视频、校准、重新预测、时间轴校对到最终输出的完整流程。内容直接来自项目中的 Markdown 文件。</DialogDescription>
        </DialogHeader>
        <ScrollArea className="tutorial-scroll">
          {error ? <div className="tutorial-error"><strong>教程载入失败</strong><span>{error}</span></div> : null}
          {!error && !markdown ? <div className="tutorial-loading"><Skeleton /><Skeleton /><Skeleton /></div> : null}
          {markdown ? <article className="markdown-doc"><Markdown>{markdown}</Markdown></article> : null}
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
}
