import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Card, CardContent } from "@/components/ui/card";
import { ResultsView } from "@/components/results/ResultsView";
import { formatDate } from "@/lib/utils";
import { api } from "@/services/api";

export function HistoryDetailPage() {
  const { id } = useParams<{ id: string }>();

  const { data, isLoading, isError } = useQuery({
    queryKey: ["history", id],
    queryFn: () => api.getHistoryItem(id!),
    enabled: Boolean(id),
  });

  return (
    <div className="space-y-6">
      <Button variant="ghost" size="sm" asChild className="gap-1.5 px-2">
        <Link to="/history">
          <ArrowLeft className="h-4 w-4" />
          Back to history
        </Link>
      </Button>

      {isLoading && (
        <div className="space-y-4">
          <Skeleton className="h-24 w-full" />
          <Skeleton className="h-64 w-full" />
        </div>
      )}

      {isError && (
        <Card>
          <CardContent className="py-8 text-center text-sm text-muted-foreground">
            Couldn't load this record.
          </CardContent>
        </Card>
      )}

      {data && (
        <>
          <div>
            <h1 className="text-xl font-semibold">Screening detail</h1>
            <p className="text-sm text-muted-foreground">
              Recorded {formatDate(data.created_at)} · model {data.model_version}
            </p>
          </div>
          <ResultsView response={data.full_response} />
        </>
      )}
    </div>
  );
}
