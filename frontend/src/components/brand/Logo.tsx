interface LogoMarkProps {
  className?: string;
}

let gradientId = 0;

export function LogoMark({ className }: LogoMarkProps) {
  const id = `logo-grad-${gradientId++}`;
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke={`url(#${id})`}
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
    >
      <defs>
        <linearGradient id={id} x1="2" y1="2" x2="22" y2="22" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="hsl(var(--primary))" />
          <stop offset="1" stopColor="hsl(var(--brand-accent))" />
        </linearGradient>
      </defs>
      <path d="M6.3 15.3C5.1 14.3 4.5 13 4.5 11.5 4.5 7.9 7.6 5 11.5 5s9 3.4 9.5 7.9" />
      <path d="M8.6 17.3c.9-.3 1.4-1.1 1.4-2v-3.1c0-1.7 1.4-3.1 3.1-3.1.9 0 1.7.4 2.2.9" />
      <circle cx="19.5" cy="13.1" r="0.9" fill={`url(#${id})`} stroke="none" />
      <circle cx="9.7" cy="19" r="0.9" fill={`url(#${id})`} stroke="none" />
    </svg>
  );
}

interface LogoProps {
  iconClassName?: string;
  titleClassName?: string;
  subtitleClassName?: string;
  showSubtitle?: boolean;
}

export function Logo({ iconClassName = "h-8 w-8", titleClassName, subtitleClassName, showSubtitle = true }: LogoProps) {
  return (
    <div className="flex items-center gap-2.5">
      <LogoMark className={iconClassName} />
      <div>
        <div className={titleClassName ?? "text-sm font-semibold leading-none tracking-tight"}>
          NEUROSCREEN <span className="text-gradient-brand">AI</span>
        </div>
        {showSubtitle && (
          <div className={subtitleClassName ?? "mt-1 text-[11px] leading-none text-muted-foreground"}>
            Research prototype
          </div>
        )}
      </div>
    </div>
  );
}
