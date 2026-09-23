import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { ArrowLeft, ArrowRight, CheckCircle2, Loader2, RotateCcw } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { ScoreToggle } from "@/components/screening/ScoreToggle";
import { StepProgressBar } from "@/components/screening/StepProgressBar";
import { VideoObservationPanel } from "@/components/video/VideoObservationPanel";
import { ResultsView } from "@/components/results/ResultsView";
import { api, ApiError } from "@/services/api";
import { ETHNICITY_OPTIONS, GENDER_OPTIONS, RELATION_OPTIONS, YES_NO_OPTIONS } from "@/lib/formOptions";
import { AQ10_QUESTIONS } from "@/lib/aq10Questions";
import type { AssessmentRequest, AssessmentResponse } from "@/types/api";

const EMPTY_FORM: AssessmentRequest = {
  A1_Score: null,
  A2_Score: null,
  A3_Score: null,
  A4_Score: null,
  A5_Score: null,
  A6_Score: null,
  A7_Score: null,
  A8_Score: null,
  A9_Score: null,
  A10_Score: null,
  age: null,
  gender: null,
  ethnicity: null,
  jundice: null,
  austim: null,
  contry_of_res: null,
  used_app_before: null,
  relation: null,
};

const SCORE_KEYS = [
  "A1_Score",
  "A2_Score",
  "A3_Score",
  "A4_Score",
  "A5_Score",
  "A6_Score",
  "A7_Score",
  "A8_Score",
  "A9_Score",
  "A10_Score",
] as const satisfies readonly (keyof AssessmentRequest)[];

const STEP_COUNT = 3;

export function ScreeningPage() {
  const [step, setStep] = useState(0);
  const [form, setForm] = useState<AssessmentRequest>(EMPTY_FORM);
  const [result, setResult] = useState<AssessmentResponse | null>(null);

  const mutation = useMutation({
    mutationFn: (payload: AssessmentRequest) => api.createAssessment(payload),
    onSuccess: (data) => {
      setResult(data);
      toast.success("Screening complete");
      window.scrollTo({ top: 0, behavior: "smooth" });
    },
    onError: (err: unknown) => {
      const message = err instanceof ApiError ? err.message : "Something went wrong.";
      toast.error("Screening failed", { description: message });
    },
  });

  const allScoresAnswered = SCORE_KEYS.every((key) => form[key] !== null);
  const characteristicsValid = form.age !== null && form.gender !== null;
  const canSubmit = allScoresAnswered && characteristicsValid && !mutation.isPending;

  function update<K extends keyof AssessmentRequest>(key: K, value: AssessmentRequest[K]) {
    setForm((f) => ({ ...f, [key]: value }));
  }

  function updateScore(key: (typeof SCORE_KEYS)[number], value: number) {
    setForm((f) => ({ ...f, [key]: value }));
  }

  function goNext() {
    if (step === 0 && !allScoresAnswered) {
      toast.error("Please answer all 10 screening items before continuing.");
      return;
    }
    setStep((s) => Math.min(s + 1, STEP_COUNT - 1));
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function goBack() {
    setStep((s) => Math.max(s - 1, 0));
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!canSubmit) {
      toast.error("Please answer all 10 screening items, age, and gender before submitting.");
      return;
    }
    mutation.mutate(form);
  }

  function handleReset() {
    setForm(EMPTY_FORM);
    setResult(null);
    setStep(0);
  }

  if (result) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-semibold">Screening result</h1>
            <p className="text-sm text-muted-foreground">Research/educational screening-support output.</p>
          </div>
          <Button variant="outline" onClick={handleReset} className="gap-2">
            <RotateCcw className="h-4 w-4" />
            New screening
          </Button>
        </div>
        <ResultsView response={result} />
      </div>
    );
  }

  return (
    <form onSubmit={handleSubmit} className="mx-auto max-w-3xl space-y-6">
      <div className="space-y-4">
        <div>
          <h1 className="text-xl font-semibold">New screening</h1>
          <p className="text-sm text-muted-foreground">
            Three quick steps — behavioral items, optional video, and background context.
          </p>
        </div>
        <StepProgressBar total={STEP_COUNT} current={step} />
      </div>

      {step === 0 && (
        <Card className="card-elevated animate-fade-in xl:-mx-[10%]">
          <CardHeader>
            <CardTitle>Behavioral screening items</CardTitle>
            <CardDescription>
              Ten binary items, worded per the Q-CHAT-10-Toddler instrument (the source with
              documented item text). For ages 1–3 these are scored by the toddler model as-is;
              for ages 4–11 the same yes/no answers are scored by a separate AQ-10-Child model
              whose exact item wording isn't available in this dataset, so treat the phrasing
              below as a close proxy for that age group, not a verbatim instrument match.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            {SCORE_KEYS.map((key, i) => (
              <ScoreToggle
                key={key}
                label={`${AQ10_QUESTIONS[i].code} — ${AQ10_QUESTIONS[i].title}`}
                description={AQ10_QUESTIONS[i].question}
                value={form[key]}
                onChange={(v) => updateScore(key, v)}
              />
            ))}
          </CardContent>
        </Card>
      )}

      {step === 1 && (
        <div className="animate-fade-in">
          <VideoObservationPanel />
        </div>
      )}

      {step === 2 && (
        <Card className="card-elevated animate-fade-in">
          <CardHeader>
            <CardTitle>Individual characteristics</CardTitle>
            <CardDescription>Demographic and background fields used by the model.</CardDescription>
          </CardHeader>
          <CardContent className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div className="space-y-1.5">
              <Label htmlFor="age">Age (years)</Label>
              <Input
                id="age"
                type="number"
                min={0}
                max={120}
                step="any"
                value={form.age ?? ""}
                onChange={(e) => update("age", e.target.value === "" ? null : Number(e.target.value))}
                placeholder="e.g. 8"
              />
              <p className="text-xs text-muted-foreground">
                Ages 1–3 are scored by the toddler model, ages 4–11 by the child model. Ages
                outside both ranges (under 1, 3–4, or over 11) are still accepted but will be
                flagged as unreliable (routed for review) rather than scored as if valid.
              </p>
            </div>

            <div className="space-y-1.5">
              <Label>Gender</Label>
              <Select value={form.gender ?? undefined} onValueChange={(v) => update("gender", v)}>
                <SelectTrigger>
                  <SelectValue placeholder="Select gender" />
                </SelectTrigger>
                <SelectContent>
                  {GENDER_OPTIONS.map((o) => (
                    <SelectItem key={o.value} value={o.value}>
                      {o.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-1.5">
              <Label>Ethnicity</Label>
              <Select value={form.ethnicity ?? undefined} onValueChange={(v) => update("ethnicity", v)}>
                <SelectTrigger>
                  <SelectValue placeholder="Select ethnicity" />
                </SelectTrigger>
                <SelectContent>
                  {ETHNICITY_OPTIONS.map((o) => (
                    <SelectItem key={o} value={o}>
                      {o}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-1.5">
              <Label>Country of residence</Label>
              <Input
                value={form.contry_of_res ?? ""}
                onChange={(e) => update("contry_of_res", e.target.value || null)}
                placeholder="e.g. United States"
              />
            </div>

            <div className="space-y-1.5">
              <Label>Born with jaundice</Label>
              <Select value={form.jundice ?? undefined} onValueChange={(v) => update("jundice", v)}>
                <SelectTrigger>
                  <SelectValue placeholder="Select" />
                </SelectTrigger>
                <SelectContent>
                  {YES_NO_OPTIONS.map((o) => (
                    <SelectItem key={o.value} value={o.value}>
                      {o.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-1.5">
              <Label>Family member with autism</Label>
              <Select value={form.austim ?? undefined} onValueChange={(v) => update("austim", v)}>
                <SelectTrigger>
                  <SelectValue placeholder="Select" />
                </SelectTrigger>
                <SelectContent>
                  {YES_NO_OPTIONS.map((o) => (
                    <SelectItem key={o.value} value={o.value}>
                      {o.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-1.5">
              <Label>Used a screening app before</Label>
              <Select value={form.used_app_before ?? undefined} onValueChange={(v) => update("used_app_before", v)}>
                <SelectTrigger>
                  <SelectValue placeholder="Select" />
                </SelectTrigger>
                <SelectContent>
                  {YES_NO_OPTIONS.map((o) => (
                    <SelectItem key={o.value} value={o.value}>
                      {o.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-1.5">
              <Label>Relation to the child</Label>
              <Select value={form.relation ?? undefined} onValueChange={(v) => update("relation", v)}>
                <SelectTrigger>
                  <SelectValue placeholder="Select" />
                </SelectTrigger>
                <SelectContent>
                  {RELATION_OPTIONS.map((o) => (
                    <SelectItem key={o} value={o}>
                      {o}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </CardContent>
        </Card>
      )}

      <Separator />

      <div className="flex items-center justify-between gap-3">
        {step > 0 ? (
          <Button type="button" variant="outline" onClick={goBack} className="gap-2">
            <ArrowLeft className="h-4 w-4" />
            Back
          </Button>
        ) : (
          <span />
        )}

        {step < STEP_COUNT - 1 ? (
          <Button type="button" onClick={goNext} className="gap-2">
            Continue
            <ArrowRight className="h-4 w-4" />
          </Button>
        ) : (
          <Button type="submit" disabled={!canSubmit} className="gap-2">
            {mutation.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <CheckCircle2 className="h-4 w-4" />}
            Run screening
          </Button>
        )}
      </div>

      {step === STEP_COUNT - 1 && (
        <p className="text-center text-xs text-muted-foreground">
          This tool does not provide a diagnosis. Results should be interpreted by a qualified professional.
        </p>
      )}
    </form>
  );
}
