import { useQuery } from "@tanstack/react-query";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import {
  AlertTriangle,
  CheckCircle2,
  Gauge,
  Info,
  ListChecks,
  TrendingUp,
  type LucideIcon,
} from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { formatDate, formatPercent } from "@/lib/utils";
import { GENDER_OPTIONS } from "@/lib/formOptions";
import { api } from "@/services/api";

const GENDER_LABELS = Object.fromEntries(GENDER_OPTIONS.map((o) => [o.value.toLowerCase(), o.label]));

function formatSubgroupValue(field: string, value: string): string {
  if (field.toLowerCase() === "gender") return GENDER_LABELS[value.toLowerCase()] ?? value;
  return value;
}

function StatTile({ label, value, icon: Icon }: { label: string; value: string; icon?: LucideIcon }) {
  return (
    <div className="card-elevated rounded-lg border p-4">
      <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
        {Icon && <Icon className="h-3.5 w-3.5" />}
        {label}
      </div>
      <div className="mt-1.5 text-2xl font-semibold tabular-nums">{value}</div>
    </div>
  );
}

function toChartData(dist: Record<string, number>) {
  return Object.entries(dist).map(([name, value]) => ({ name, value }));
}

function PerformanceTab() {
  const { data, isLoading } = useQuery({
    queryKey: ["monitoring", "performance"],
    queryFn: api.monitoring.performance,
  });

  if (isLoading) return <Skeleton className="h-64 w-full" />;
  if (!data || data.total_screenings === 0) {
    return <p className="py-8 text-center text-sm text-muted-foreground">No screenings recorded yet.</p>;
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <StatTile label="Total screenings" value={String(data.total_screenings)} icon={ListChecks} />
        <StatTile label="Avg. probability (YES)" value={formatPercent(data.avg_probability_yes)} icon={TrendingUp} />
        <StatTile
          label="Groundedness pass rate"
          value={formatPercent(data.groundedness_pass_rate)}
          icon={CheckCircle2}
        />
        <StatTile label="Fallback rate" value={formatPercent(data.fallback_rate)} icon={Gauge} />
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Card className="card-elevated">
          <CardHeader>
            <CardTitle className="text-sm">Route distribution</CardTitle>
          </CardHeader>
          <CardContent className="h-52">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={toChartData(data.route_distribution)}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} stroke="hsl(var(--muted-foreground))" />
                <YAxis allowDecimals={false} tick={{ fontSize: 11 }} stroke="hsl(var(--muted-foreground))" />
                <Tooltip
                  contentStyle={{
                    background: "hsl(var(--popover))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                />
                <Bar dataKey="value" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card className="card-elevated">
          <CardHeader>
            <CardTitle className="text-sm">Prediction label distribution</CardTitle>
          </CardHeader>
          <CardContent className="h-52">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={toChartData(data.label_distribution)}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} stroke="hsl(var(--muted-foreground))" />
                <YAxis allowDecimals={false} tick={{ fontSize: 11 }} stroke="hsl(var(--muted-foreground))" />
                <Tooltip
                  contentStyle={{
                    background: "hsl(var(--popover))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                />
                <Bar dataKey="value" fill="hsl(var(--brand-accent))" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function FairnessTab() {
  const { data, isLoading } = useQuery({
    queryKey: ["monitoring", "fairness"],
    queryFn: api.monitoring.fairness,
  });

  if (isLoading) return <Skeleton className="h-64 w-full" />;
  if (!data) return null;

  return (
    <div className="space-y-4">
      <Alert variant="warning">
        <AlertTriangle className="h-4 w-4" />
        <AlertDescription>{data.disclaimer}</AlertDescription>
      </Alert>

      {Object.entries(data.breakdown).map(([field, groups]) => (
        <Card key={field}>
          <CardHeader>
            <CardTitle className="text-sm capitalize">{field}</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {Object.entries(groups).map(([value, stats]) => (
              <div key={value} className="flex items-center justify-between rounded-md border px-3 py-2 text-sm">
                <span>{formatSubgroupValue(field, value)}</span>
                <div className="flex items-center gap-3 text-xs text-muted-foreground">
                  <span>n={stats.n}</span>
                  <Badge variant="outline">{formatPercent(stats.positive_rate)} positive</Badge>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

function DriftTab() {
  const { data, isLoading } = useQuery({
    queryKey: ["monitoring", "drift"],
    queryFn: api.monitoring.drift,
  });

  if (isLoading) return <Skeleton className="h-64 w-full" />;
  if (!data) return null;

  return (
    <div className="space-y-4">
      {(data.ood_drift_detected || data.prediction_drift_detected) && (
        <Alert variant="warning">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            Drift signal detected between recent traffic and full history — worth investigating.
          </AlertDescription>
        </Alert>
      )}
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <StatTile label="OOD rate (recent)" value={formatPercent(data.ood_rate_recent)} />
        <StatTile label="OOD rate (overall)" value={formatPercent(data.ood_rate_overall)} />
        <StatTile label="Avg. P(yes) (recent)" value={formatPercent(data.avg_probability_recent)} />
        <StatTile label="Avg. P(yes) (overall)" value={formatPercent(data.avg_probability_overall)} />
      </div>
      <Alert>
        <Info className="h-4 w-4" />
        <AlertDescription>
          Feature/population drift is approximated by the model's own out-of-distribution flag rate;
          prediction drift compares recent vs. all-time average positive-class probability. Directional
          signals only — there's no ground truth for live traffic.
        </AlertDescription>
      </Alert>
    </div>
  );
}

function AuditTab() {
  const { data, isLoading } = useQuery({
    queryKey: ["monitoring", "audit"],
    queryFn: api.monitoring.audit,
  });

  if (isLoading) return <Skeleton className="h-64 w-full" />;
  if (!data || data.entries.length === 0) {
    return <p className="py-8 text-center text-sm text-muted-foreground">No audit entries yet.</p>;
  }

  return (
    <div className="card-elevated overflow-x-auto rounded-lg border">
      <table className="w-full text-sm">
        <thead className="bg-muted/50 text-xs text-muted-foreground">
          <tr>
            <th className="px-3 py-2 text-left font-medium">Timestamp</th>
            <th className="px-3 py-2 text-left font-medium">Route</th>
            <th className="px-3 py-2 text-left font-medium">Label</th>
            <th className="px-3 py-2 text-left font-medium">Groundedness</th>
            <th className="px-3 py-2 text-left font-medium">Audit</th>
          </tr>
        </thead>
        <tbody>
          {data.entries.map((entry) => (
            <tr key={entry.id} className="border-t border-border/60 hover:bg-accent/20">
              <td className="px-3 py-2 text-xs text-muted-foreground">{formatDate(entry.timestamp)}</td>
              <td className="px-3 py-2">{entry.route.replace(/_/g, " ")}</td>
              <td className="px-3 py-2">{entry.prediction_label}</td>
              <td className="px-3 py-2">{formatPercent(entry.groundedness_score)}</td>
              <td className="px-3 py-2">
                <Badge variant={entry.audit_status === "PASSED" ? "success" : "warning"}>
                  {entry.audit_status}
                </Badge>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function MonitoringPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold">Monitoring</h1>
        <CardDescription>
          Live-traffic monitoring over recorded screenings — not an offline model evaluation report.
        </CardDescription>
      </div>

      <Tabs defaultValue="performance">
        <TabsList>
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="fairness">Fairness</TabsTrigger>
          <TabsTrigger value="drift">Drift</TabsTrigger>
          <TabsTrigger value="audit">Audit trail</TabsTrigger>
        </TabsList>
        <TabsContent value="performance">
          <PerformanceTab />
        </TabsContent>
        <TabsContent value="fairness">
          <FairnessTab />
        </TabsContent>
        <TabsContent value="drift">
          <DriftTab />
        </TabsContent>
        <TabsContent value="audit">
          <AuditTab />
        </TabsContent>
      </Tabs>
    </div>
  );
}
