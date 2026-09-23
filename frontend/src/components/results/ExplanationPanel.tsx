import { AlertTriangle, Info } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { AssessmentResponse } from "@/types/api";

const CITATION_RE = /\s*\[E\d+\]/gi;

/** Strip inline [E1]-style citation markers from the displayed prose --
 * every claim is still individually cited under the hood (see the
 * Groundedness tab for the claim-by-claim breakdown and the Evidence
 * tab for the underlying sources), but showing bracket codes in the
 * middle of sentences reads as noise rather than a citation to most
 * readers. */
function stripCitations(text: string): string {
  return text.replace(CITATION_RE, "").replace(/\s+([,.!?])/g, "$1").trim();
}

const FALLBACK_MESSAGES: Record<string, string> = {
  deterministic_grounded_fallback:
    "The model's generated explanation didn't pass groundedness checks, so this is a deterministic, evidence-quoted fallback explanation instead.",
  deterministic_safe_fallback:
    "No LLM explanation was available (and no evidence could be retrieved to quote), so this is a fully deterministic safe fallback response.",
};

const DEFAULT_FALLBACK_MESSAGE =
  "The model's own explanation didn't pass safety checks, so a deterministic fallback is shown instead.";

export function ExplanationPanel({ response }: { response: AssessmentResponse }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Explanation</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {response.fallback_used && (
          <Alert variant="warning">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>
              {FALLBACK_MESSAGES[response.generator] ?? DEFAULT_FALLBACK_MESSAGE}
            </AlertDescription>
          </Alert>
        )}

        <p className="whitespace-pre-wrap text-sm leading-relaxed">
          {stripCitations(response.llm_explanation)}
        </p>

        {response.citations.length > 0 && (
          <p className="text-xs text-muted-foreground">
            Based on {response.citations.length} source{response.citations.length === 1 ? "" : "s"} —
            see the Evidence tab for details.
          </p>
        )}

        <div className="text-xs text-muted-foreground">Generator: {response.generator}</div>

        <Alert>
          <Info className="h-4 w-4" />
          <AlertDescription>{response.disclaimer}</AlertDescription>
        </Alert>
      </CardContent>
    </Card>
  );
}
