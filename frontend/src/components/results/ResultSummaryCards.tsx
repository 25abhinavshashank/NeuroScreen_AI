import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { formatPercent } from "@/lib/utils";
import type { AssessmentResponse } from "@/types/api";
import {
  DecisionBadge,
  DistributionBadge,
  PredictionBadge,
  UncertaintyBadge,
} from "./StatusBadges";

const AGE_BAND_LABEL: Record<string, string> = {
  toddler: "Toddler model (Q-CHAT-10, 1–3yr)",
  child: "Child model (AQ-10, 4–11yr)",
};

const AGE_BAND_RANGE: Record<string, string> = {
  toddler: "1–3yr training range",
  child: "4–11yr training range",
};

export function ResultSummaryCards({ response }: { response: AssessmentResponse }) {
  const { prediction, model, uncertainty, distribution, decision } = response;

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-xs font-medium text-muted-foreground">Model prediction</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <PredictionBadge label={prediction.label} />
          <div className="text-2xl font-semibold tabular-nums">
            {formatPercent(prediction.probability_yes)}
          </div>
          <p className="text-xs text-muted-foreground">
            {AGE_BAND_LABEL[model.age_band] ?? `${model.name} ${model.version}`}
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-xs font-medium text-muted-foreground">Uncertainty</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <UncertaintyBadge category={uncertainty.category} />
          <div className="text-2xl font-semibold tabular-nums">{formatPercent(uncertainty.confidence)}</div>
          <Progress value={uncertainty.confidence * 100} className="h-1.5" />
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-xs font-medium text-muted-foreground">Distribution check</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <DistributionBadge status={distribution.status} />
          <div className="text-2xl font-semibold tabular-nums">
            {distribution.distance !== null ? distribution.distance.toFixed(3) : "—"}
          </div>
          <p className="text-xs text-muted-foreground">
            {distribution.age_out_of_range
              ? `Age outside the ${AGE_BAND_RANGE[model.age_band] ?? "model's training range"}`
              : `Distance vs. threshold ${distribution.threshold?.toFixed(3) ?? "—"}`}
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-xs font-medium text-muted-foreground">Workflow decision</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <DecisionBadge status={decision.status} />
          <p className="text-xs leading-snug text-muted-foreground">{decision.reason}</p>
        </CardContent>
      </Card>
    </div>
  );
}
