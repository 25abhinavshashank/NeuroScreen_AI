import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ShieldAlert } from "lucide-react";
import type { AssessmentResponse } from "@/types/api";
import { ResultSummaryCards } from "./ResultSummaryCards";
import { ExplanationPanel } from "./ExplanationPanel";
import { EvidencePanel } from "./EvidencePanel";
import { GroundednessPanel } from "./GroundednessPanel";
import { AuditPanel } from "./AuditPanel";
import { RouteBadge, SafetyBadge } from "./StatusBadges";

export function ResultsView({ response }: { response: AssessmentResponse }) {
  const isHumanOrOod = response.route === "HUMAN_REVIEW" || response.route === "OOD_REVIEW";

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-2">
        <RouteBadge route={response.route} />
        <SafetyBadge status={response.safety_status} />
      </div>

      {isHumanOrOod && (
        <Alert variant="warning">
          <ShieldAlert className="h-4 w-4" />
          <AlertTitle>This result was routed for additional review</AlertTitle>
          <AlertDescription>{response.route_reason}</AlertDescription>
        </Alert>
      )}

      <p className="text-sm leading-relaxed text-muted-foreground">{response.summary}</p>

      <ResultSummaryCards response={response} />

      <Tabs defaultValue="explanation">
        <TabsList>
          <TabsTrigger value="explanation">Explanation</TabsTrigger>
          <TabsTrigger value="evidence">Evidence ({response.evidence.length})</TabsTrigger>
          <TabsTrigger value="groundedness">Groundedness</TabsTrigger>
          <TabsTrigger value="audit">Audit</TabsTrigger>
        </TabsList>

        <TabsContent value="explanation">
          <ExplanationPanel response={response} />
        </TabsContent>

        <TabsContent value="evidence">
          <EvidencePanel evidence={response.evidence} citations={response.citations} />
        </TabsContent>

        <TabsContent value="groundedness">
          <GroundednessPanel groundedness={response.groundedness} />
        </TabsContent>

        <TabsContent value="audit">
          <AuditPanel audit={response.audit} />
        </TabsContent>
      </Tabs>
    </div>
  );
}
