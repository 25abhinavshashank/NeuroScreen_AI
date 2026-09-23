import { CheckCircle2, XCircle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { cn, formatPercent } from "@/lib/utils";
import type { GroundednessResult } from "@/types/api";

function Metric({ label, value }: { label: string; value: number }) {
  return (
    <div className="space-y-1">
      <div className="flex items-baseline justify-between">
        <span className="text-xs text-muted-foreground">{label}</span>
        <span className="text-sm font-medium tabular-nums">{formatPercent(value)}</span>
      </div>
      <Progress value={value * 100} className="h-1.5" />
    </div>
  );
}

export function GroundednessPanel({ groundedness }: { groundedness: GroundednessResult }) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>Groundedness</CardTitle>
          <Badge variant={groundedness.passed ? "success" : "destructive"}>
            {groundedness.passed ? "Passed" : "Failed"}
          </Badge>
        </div>
        <CardDescription>
          System-quality metrics on how well the explanation is supported by cited evidence — not a
          measure of clinical reliability.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <Metric label="Groundedness score" value={groundedness.groundedness_score} />
          <Metric label="Citation coverage" value={groundedness.citation_coverage} />
          <Metric label="Citation precision" value={groundedness.citation_precision} />
          <Metric label="Unsupported claim rate" value={groundedness.unsupported_claim_rate} />
        </div>

        <div className="flex gap-4 text-xs text-muted-foreground">
          <span>{groundedness.total_claims} claims</span>
          <span>{groundedness.supported_claims} supported</span>
          <span>{groundedness.cited_claims} cited</span>
        </div>

        {groundedness.claims.length > 0 && (
          <>
            <Separator />
            <div className="space-y-2">
              {groundedness.claims.map((claim, i) => (
                <div key={i} className="flex items-start gap-2 rounded-md border p-2.5">
                  {claim.supported ? (
                    <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-success" />
                  ) : (
                    <XCircle className="mt-0.5 h-4 w-4 shrink-0 text-destructive" />
                  )}
                  <div className="min-w-0 flex-1">
                    <p className={cn("text-xs leading-relaxed", !claim.supported && "text-muted-foreground")}>
                      {claim.claim}
                    </p>
                    <div className="mt-1 flex flex-wrap gap-1">
                      {claim.citation_ids.map((id) => (
                        <Badge key={id} variant="outline" className="text-[10px]">
                          {id}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
