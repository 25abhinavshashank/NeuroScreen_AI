import { Badge } from "@/components/ui/badge";
import type { DecisionStatus, DistributionStatus, RouteType, SafetyStatus, UncertaintyCategory } from "@/types/api";

export function RouteBadge({ route }: { route: RouteType }) {
  const variant =
    route === "STANDARD_REVIEW"
      ? "success"
      : route === "HUMAN_REVIEW"
        ? "warning"
        : route === "OOD_REVIEW"
          ? "warning"
          : "destructive";
  const label = route.replace(/_/g, " ");
  return <Badge variant={variant}>{label}</Badge>;
}

export function SafetyBadge({ status }: { status: SafetyStatus }) {
  return (
    <Badge variant={status === "SCREENING_ONLY" ? "secondary" : "destructive"}>
      {status.replace(/_/g, " ")}
    </Badge>
  );
}

export function UncertaintyBadge({ category }: { category: UncertaintyCategory }) {
  const variant = category === "LOW" ? "success" : category === "MEDIUM" ? "warning" : "destructive";
  return <Badge variant={variant}>{category} uncertainty</Badge>;
}

export function DistributionBadge({ status }: { status: DistributionStatus }) {
  const variant =
    status === "IN_DISTRIBUTION" ? "success" : status === "OUT_OF_DISTRIBUTION" ? "destructive" : "outline";
  return <Badge variant={variant}>{status.replace(/_/g, " ")}</Badge>;
}

export function DecisionBadge({ status }: { status: DecisionStatus }) {
  const variant = status === "SCREENING_RESULT" ? "success" : "warning";
  return <Badge variant={variant}>{status.replace(/_/g, " ")}</Badge>;
}

export function PredictionBadge({ label }: { label: string }) {
  return (
    <Badge variant={label === "YES" ? "warning" : "success"} className="text-sm px-3 py-1">
      {label === "YES" ? "Positive screen" : "Negative screen"}
    </Badge>
  );
}
