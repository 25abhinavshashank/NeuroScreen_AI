import { FileText } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import type { Citation, EvidenceItem } from "@/types/api";

export function EvidencePanel({ evidence, citations }: { evidence: EvidenceItem[]; citations: Citation[] }) {
  const items = evidence.length > 0 ? evidence : citations;

  if (items.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Evidence</CardTitle>
          <CardDescription>No supporting evidence was retrieved for this query.</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Retrieved evidence</CardTitle>
        <CardDescription>
          Knowledge-base passages retrieved and reranked for this query, cited in the explanation above.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {items.map((item, i) => (
          <div key={item.chunk_id ?? i} className="rounded-lg border p-3">
            <div className="mb-1.5 flex flex-wrap items-center gap-2">
              <Badge variant="secondary">{item.citation_id ?? `[E${i + 1}]`}</Badge>
              <span className="flex items-center gap-1 text-xs text-muted-foreground">
                <FileText className="h-3 w-3" />
                {item.file_name ?? "unknown source"}
              </span>
              {"similarity" in item && item.similarity != null && (
                <span className="text-xs text-muted-foreground">
                  similarity {item.similarity.toFixed(3)}
                </span>
              )}
            </div>
            <p className="text-sm leading-relaxed text-foreground/90">{item.text}</p>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
