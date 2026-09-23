import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { ChevronRight, Inbox } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { PredictionBadge, RouteBadge } from "@/components/results/StatusBadges";
import { Badge } from "@/components/ui/badge";
import { cn, formatDate, formatPercent } from "@/lib/utils";
import { api } from "@/services/api";

export function HistoryPage() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["history"],
    queryFn: () => api.listHistory({ limit: 50 }),
  });

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold">Screening history</h1>
        <p className="text-sm text-muted-foreground">Past screenings recorded by this instance.</p>
      </div>

      {isLoading && (
        <div className="space-y-3">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-20 w-full" />
          ))}
        </div>
      )}

      {isError && (
        <Card>
          <CardContent className="py-8 text-center text-sm text-muted-foreground">
            Couldn't load history. Is the backend running?
          </CardContent>
        </Card>
      )}

      {data && data.items.length === 0 && (
        <Card>
          <CardContent className="flex flex-col items-center gap-2 py-12 text-center text-muted-foreground">
            <Inbox className="h-8 w-8" />
            <p className="text-sm">No screenings recorded yet.</p>
          </CardContent>
        </Card>
      )}

      {data && data.items.length > 0 && (
        <div className="space-y-2.5">
          {data.items.map((item) => (
            <Link key={item.id} to={`/history/${item.id}`}>
              <Card
                className={cn(
                  "card-elevated group border-l-[3px] transition-all hover:-translate-y-0.5 hover:border-primary/40",
                  item.prediction_label === "YES" ? "border-l-warning" : "border-l-success",
                )}
              >
                <CardContent className="flex flex-wrap items-center justify-between gap-3 py-4">
                  <div className="flex flex-wrap items-center gap-2.5">
                    <PredictionBadge label={item.prediction_label} />
                    <RouteBadge route={item.route} />
                    {!item.groundedness_passed && <Badge variant="destructive">Groundedness failed</Badge>}
                    <span className="text-sm text-muted-foreground">
                      {formatPercent(item.probability_yes)} probability
                    </span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="hidden text-xs text-muted-foreground sm:inline">
                      {formatDate(item.created_at)}
                    </span>
                    <ChevronRight className="h-4 w-4 text-muted-foreground transition-transform group-hover:translate-x-0.5 group-hover:text-foreground" />
                  </div>
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
