import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { Slot } from "radix-ui"

import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "group/button inline-flex shrink-0 items-center justify-center rounded-[10px] border border-transparent bg-clip-padding text-[13px] font-medium whitespace-nowrap transition-[color,background-color,border-color,transform] duration-150 ease-[cubic-bezier(.16,1,.3,1)] outline-none select-none focus-visible:ring-3 focus-visible:ring-ring/30 active:not-aria-[haspopup]:scale-[0.97] disabled:pointer-events-none disabled:opacity-45 aria-invalid:border-destructive aria-invalid:ring-3 aria-invalid:ring-destructive/20 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4",
  {
    variants: {
      variant: {
        default:
          "bg-primary text-primary-foreground shadow-[inset_0_1px_0_rgb(255_255_255/.16),var(--shadow-xs)] hover:bg-(--brand-hover)",
        outline:
          "border-(--line-strong) bg-(--panel) text-foreground shadow-(--shadow-xs) hover:border-(--brand-line) hover:bg-(--brand-softer) hover:text-(--brand-strong) aria-expanded:border-(--brand-line) aria-expanded:bg-(--brand-softer)",
        secondary:
          "bg-(--brand-soft) text-(--brand-strong) hover:bg-[color-mix(in_oklab,var(--brand-soft),var(--brand)_12%)] aria-expanded:bg-(--brand-soft)",
        ghost:
          "text-muted-foreground hover:bg-(--panel-raised) hover:text-foreground aria-expanded:bg-(--panel-raised) aria-expanded:text-foreground",
        destructive:
          "bg-(--danger-soft) text-destructive hover:bg-[color-mix(in_oklab,var(--danger-soft),var(--red)_12%)] focus-visible:ring-destructive/20",
        link: "text-primary underline-offset-4 hover:underline",
      },
      size: {
        default: "h-9 gap-2 px-3.5 has-data-[icon=inline-end]:pr-3 has-data-[icon=inline-start]:pl-3",
        xs: "h-7 gap-1.5 rounded-[8px] px-2.5 text-xs [&_svg:not([class*='size-'])]:size-3.5",
        sm: "h-8 gap-1.5 rounded-[9px] px-3 [&_svg:not([class*='size-'])]:size-4",
        lg: "h-10 gap-2 px-4",
        icon: "size-9",
        "icon-xs": "size-7 rounded-[8px] [&_svg:not([class*='size-'])]:size-3.5",
        "icon-sm": "size-8 rounded-[9px]",
        "icon-lg": "size-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

function Button({
  className,
  variant = "default",
  size = "default",
  asChild = false,
  ...props
}: React.ComponentProps<"button"> &
  VariantProps<typeof buttonVariants> & {
    asChild?: boolean
  }) {
  const Comp = asChild ? Slot.Root : "button"

  return (
    <Comp
      data-slot="button"
      data-variant={variant}
      data-size={size}
      className={cn(buttonVariants({ variant, size, className }))}
      {...props}
    />
  )
}

export { Button, buttonVariants }
