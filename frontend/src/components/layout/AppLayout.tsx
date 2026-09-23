import { useEffect, useState } from "react";
import { Link, NavLink, Outlet } from "react-router-dom";
import { ClipboardList, History, LayoutDashboard, Menu, Moon, PanelLeftClose, Sun } from "lucide-react";
import { cn } from "@/lib/utils";
import { useTheme } from "@/hooks/useTheme";
import { Button } from "@/components/ui/button";
import { Logo, LogoMark } from "@/components/brand/Logo";

const navItems = [
  { to: "/screening", label: "Screening", icon: ClipboardList, end: true },
  { to: "/history", label: "History", icon: History },
  { to: "/monitoring", label: "Monitoring", icon: LayoutDashboard },
];

const SIDEBAR_COLLAPSED_KEY = "neuroscreen-sidebar-collapsed";

function getInitialCollapsed(): boolean {
  try {
    return localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === "1";
  } catch {
    return false;
  }
}

export function AppLayout() {
  const { theme, toggle } = useTheme();
  const [collapsed, setCollapsed] = useState(getInitialCollapsed);

  useEffect(() => {
    try {
      localStorage.setItem(SIDEBAR_COLLAPSED_KEY, collapsed ? "1" : "0");
    } catch {
      // ignore storage access errors
    }
  }, [collapsed]);

  return (
    <div className="flex min-h-screen bg-background">
      <aside
        className={cn(
          "sticky top-0 hidden h-screen shrink-0 flex-col overflow-y-auto border-r border-border/60 bg-card/30 py-6 transition-[width] duration-200 md:flex",
          collapsed ? "w-[76px] px-2.5" : "w-64 px-4",
        )}
      >
        <div className={cn("mb-8 flex items-center gap-2", collapsed ? "flex-col-reverse justify-center px-0" : "justify-between px-2")}>
          {collapsed ? (
            <Link to="/">
              <LogoMark className="h-8 w-8" />
            </Link>
          ) : (
            <Link to="/">
              <Logo iconClassName="h-8 w-8" />
            </Link>
          )}
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setCollapsed((c) => !c)}
            className="shrink-0 text-muted-foreground hover:text-foreground"
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {collapsed ? <Menu className="h-[18px] w-[18px]" /> : <PanelLeftClose className="h-[18px] w-[18px]" />}
          </Button>
        </div>

        <nav className="flex flex-1 flex-col gap-1">
          {navItems.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              title={collapsed ? label : undefined}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-3 rounded-md px-3 py-2.5 text-sm font-medium transition-colors",
                  collapsed && "justify-center px-0",
                  isActive
                    ? "border border-primary/20 bg-primary/10 text-foreground"
                    : "text-muted-foreground hover:bg-accent/60 hover:text-foreground",
                )
              }
            >
              <Icon className="h-5 w-5 shrink-0" />
              {!collapsed && label}
            </NavLink>
          ))}
        </nav>

        <div className={cn("mt-auto space-y-3", collapsed ? "px-0" : "px-1")}>
          <Button
            variant="outline"
            size={collapsed ? "icon" : "sm"}
            className={cn(collapsed ? "mx-auto flex" : "w-full justify-start gap-2")}
            onClick={toggle}
            title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
          >
            {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            {!collapsed && (theme === "dark" ? "Light mode" : "Dark mode")}
          </Button>
          {!collapsed && (
            <p className="px-1 text-sm leading-snug text-muted-foreground">
              Educational screening-support prototype. Not a diagnostic system.
            </p>
          )}
        </div>
      </aside>

      <div className="flex min-h-screen flex-1 flex-col">
        <header className="flex h-14 items-center justify-between border-b border-border/60 px-4 md:hidden">
          <Link to="/" className="flex items-center gap-2">
            <LogoMark className="h-7 w-7" />
            <span className="text-base font-semibold">NEUROSCREEN AI</span>
          </Link>
          <Button variant="outline" size="icon" onClick={toggle}>
            {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </Button>
        </header>

        <nav className="flex gap-1 overflow-x-auto border-b border-border/60 bg-card/30 px-3 py-2 md:hidden">
          {navItems.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) =>
                cn(
                  "flex shrink-0 items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium",
                  isActive ? "bg-primary/15 text-foreground" : "text-muted-foreground",
                )
              }
            >
              <Icon className="h-3.5 w-3.5" />
              {label}
            </NavLink>
          ))}
        </nav>

        <main className="flex-1 px-4 py-6 md:px-8 md:py-8">
          <div className="mx-auto w-full max-w-6xl">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
}
