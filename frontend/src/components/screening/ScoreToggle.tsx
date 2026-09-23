import { cn } from "@/lib/utils";

interface ScoreToggleProps {
  label: string;
  description?: string;
  value: number | null;
  onChange: (value: number) => void;
}

export function ScoreToggle({ label, description, value, onChange }: ScoreToggleProps) {
  return (
    <div className="flex items-center justify-between gap-4 rounded-lg border bg-card px-4 py-3 shadow-md shadow-black/10 dark:shadow-[0_1px_0_0_rgba(255,255,255,0.06)_inset,0_10px_28px_-8px_rgba(0,0,0,0.85)]">
      <div className="min-w-0">
        <span className="text-sm font-semibold">{label}</span>
        {description && (
          <p className="mt-1 text-sm leading-snug text-muted-foreground">{description}</p>
        )}
      </div>
      <div className="flex shrink-0 overflow-hidden rounded-lg border">
        {[
          { v: 1, text: "Yes" },
          { v: 0, text: "No" },
        ].map((opt) => (
          <button
            key={opt.v}
            type="button"
            onClick={() => onChange(opt.v)}
            className={cn(
              "min-w-[64px] px-5 py-2.5 text-sm font-semibold transition-colors",
              value === opt.v
                ? "bg-primary text-primary-foreground"
                : "bg-background text-muted-foreground hover:bg-accent",
            )}
          >
            {opt.text}
          </button>
        ))}
      </div>
    </div>
  );
}
