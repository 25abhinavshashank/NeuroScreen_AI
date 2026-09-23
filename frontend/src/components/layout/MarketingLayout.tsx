import { useState } from "react";
import { Link, NavLink, Outlet } from "react-router-dom";
import { Menu, Moon, Sun, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Logo } from "@/components/brand/Logo";
import { useTheme } from "@/hooks/useTheme";
import { cn } from "@/lib/utils";

const NAV_LINKS = [
  { to: "/screening", label: "Screening" },
  { to: "/history", label: "History" },
  { to: "/monitoring", label: "Monitoring" },
];

export function MarketingLayout() {
  const { theme, toggle } = useTheme();
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <div className="min-h-screen bg-background">
      <header className="sticky top-0 z-40 border-b border-border/60 bg-background/80 backdrop-blur-md">
        <div className="mx-auto flex h-14 w-full max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
          <Link to="/" className="shrink-0">
            <Logo iconClassName="h-8 w-8" showSubtitle={false} titleClassName="text-base font-semibold tracking-tight" />
          </Link>

          <nav className="hidden items-center gap-1 md:flex">
            {NAV_LINKS.map((link) => (
              <NavLink
                key={link.to}
                to={link.to}
                className={({ isActive }) =>
                  cn(
                    "rounded-md px-3 py-2 text-sm font-medium text-muted-foreground transition-colors hover:text-foreground",
                    isActive && "text-foreground",
                  )
                }
              >
                {link.label}
              </NavLink>
            ))}
          </nav>

          <div className="flex items-center gap-2">
            <Button variant="ghost" size="icon" onClick={toggle} className="hidden sm:inline-flex">
              {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </Button>
            <Button asChild size="sm" className="hidden sm:inline-flex">
              <Link to="/screening">Start screening</Link>
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="md:hidden"
              onClick={() => setMenuOpen((v) => !v)}
              aria-label="Toggle menu"
            >
              {menuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
            </Button>
          </div>
        </div>

        {menuOpen && (
          <nav className="flex flex-col gap-1 border-t border-border/60 px-4 py-3 md:hidden">
            {NAV_LINKS.map((link) => (
              <NavLink
                key={link.to}
                to={link.to}
                onClick={() => setMenuOpen(false)}
                className={({ isActive }) =>
                  cn(
                    "rounded-md px-3 py-2 text-sm font-medium text-muted-foreground",
                    isActive && "bg-accent/60 text-foreground",
                  )
                }
              >
                {link.label}
              </NavLink>
            ))}
            <Button variant="outline" size="sm" className="mt-1 justify-start gap-2" onClick={toggle}>
              {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
              {theme === "dark" ? "Light mode" : "Dark mode"}
            </Button>
          </nav>
        )}
      </header>

      <main>
        <Outlet />
      </main>

      <footer className="border-t border-border/60 bg-card/30 px-4 py-14 sm:px-6 lg:px-8">
        <div className="mx-auto grid w-full max-w-7xl grid-cols-1 gap-10 sm:grid-cols-2 lg:grid-cols-[1.4fr_1fr_1fr_1fr]">
          <div>
            <Logo iconClassName="h-8 w-8" showSubtitle={false} titleClassName="text-base font-semibold tracking-tight" />
            <p className="mt-4 max-w-xs text-sm leading-relaxed text-muted-foreground">
              Evidence-grounded autism screening support, combining a behavioral questionnaire,
              optional video observation, and individual characteristics into one age-aware flow.
            </p>
          </div>

          <div>
            <h3 className="text-sm font-semibold">Product</h3>
            <ul className="mt-4 space-y-2.5 text-sm text-muted-foreground">
              <li>
                <Link to="/screening" className="transition-colors hover:text-foreground">
                  Start a screening
                </Link>
              </li>
              <li>
                <Link to="/history" className="transition-colors hover:text-foreground">
                  Screening history
                </Link>
              </li>
              <li>
                <Link to="/monitoring" className="transition-colors hover:text-foreground">
                  Live monitoring
                </Link>
              </li>
            </ul>
          </div>

          <div>
            <h3 className="text-sm font-semibold">What's inside</h3>
            <ul className="mt-4 space-y-2.5 text-sm text-muted-foreground">
              <li>Behavioral screening items</li>
              <li>Video observation (experimental)</li>
              <li>Individual characteristics</li>
              <li>Evidence &amp; groundedness checks</li>
            </ul>
          </div>

          <div>
            <h3 className="text-sm font-semibold">Important notice</h3>
            <p className="mt-4 text-sm leading-relaxed text-muted-foreground">
              This is a research and educational prototype, not a medical device. It does not
              diagnose autism. Always consult a qualified healthcare professional for a formal
              evaluation.
            </p>
          </div>
        </div>

        <div className="mx-auto mt-12 flex w-full max-w-7xl flex-col items-center justify-between gap-3 border-t border-border/60 pt-6 text-center text-xs text-muted-foreground sm:flex-row sm:text-left">
          <p>&copy; {new Date().getFullYear()} NEUROSCREEN AI — a research/educational prototype.</p>
          <p>Not a diagnostic system. Results should always be interpreted by a qualified professional.</p>
        </div>
      </footer>
    </div>
  );
}
