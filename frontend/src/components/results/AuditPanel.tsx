import { Check, X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { AuditResult } from "@/types/api";

const CHECK_LABELS: Record<keyof AuditResult["checks"], string> = {
  non_empty: "Non-empty response",
  diagnostic_claims: "No diagnostic claims",
  probability_claims: "No overstated probability claims",
  reliability_claims: "No overstated reliability claims",
  causal_claims: "No unsupported causal claims",
  feature_claims: "No raw feature interpretation",
  individual_interpretation: "No individual-level interpretation",
  citations: "Citations valid",
  groundedness: "Groundedness threshold met",
  unsupported_claims: "Unsupported-claim rate within limit",
};

export function AuditPanel({ audit }: { audit: AuditResult }) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>Final safety audit</CardTitle>
          <Badge variant={audit.final_status === "PASSED" ? "success" : "warning"}>
            {audit.final_status.replace(/_/g, " ")}
          </Badge>
        </div>
        <CardDescription>
          Deterministic checks run on the generated explanation before it's ever shown — the LLM is not
          trusted to judge its own output.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          {(Object.entries(audit.checks) as [keyof AuditResult["checks"], boolean][]).map(([key, passed]) => (
            <div key={key} className="flex items-center gap-2 text-xs">
              {passed ? (
                <Check className="h-3.5 w-3.5 shrink-0 text-success" />
              ) : (
                <X className="h-3.5 w-3.5 shrink-0 text-destructive" />
              )}
              <span className={cn(!passed && "font-medium text-destructive")}>{CHECK_LABELS[key]}</span>
            </div>
          ))}
        </div>
        {audit.reason && (
          <p className="rounded-md bg-muted px-3 py-2 text-xs text-muted-foreground">{audit.reason}</p>
        )}
      </CardContent>
    </Card>
  );
}
