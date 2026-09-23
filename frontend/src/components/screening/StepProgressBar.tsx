import { cn } from "@/lib/utils";

interface StepProgressBarProps {
  total: number;
  current: number;
}

export function StepProgressBar({ total, current }: StepProgressBarProps) {
  return (
    <div className="flex w-full gap-1.5" role="progressbar" aria-valuenow={current + 1} aria-valuemin={1} aria-valuemax={total}>
      {Array.from({ length: total }).map((_, i) => (
        <div key={i} className="h-1.5 flex-1 overflow-hidden rounded-full bg-secondary">
          <div
            className={cn(
              "h-full rounded-full bg-gradient-to-r from-primary to-[hsl(var(--brand-accent))] transition-all duration-500 ease-out",
              i < current && "w-full",
              i === current && "w-full glow-primary",
              i > current && "w-0",
            )}
          />
        </div>
      ))}
    </div>
  );
}
