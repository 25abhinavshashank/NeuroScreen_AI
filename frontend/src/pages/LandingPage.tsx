import { Link } from "react-router-dom";
import {
  ArrowRight,
  Brain,
  ClipboardList,
  Eye,
  History,
  LayoutDashboard,
  ShieldCheck,
  Sparkle,
  Stethoscope,
  TrendingUp,
  Video,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { NeuralHero } from "@/components/brand/NeuralHero";

const ABOUT_AUTISM = [
  {
    icon: Brain,
    title: "What is autism spectrum disorder?",
    description:
      "ASD is a neurodevelopmental condition that affects how a person communicates, interacts socially, and experiences the world around them. It's called a \"spectrum\" because it looks different in every individual — some traits are subtle, others more pronounced.",
  },
  {
    icon: Eye,
    title: "Common early signs",
    description:
      "Limited eye contact, delayed speech or language, less pointing or showing interest in shared objects, a strong preference for routines, and sensitivity to sounds, textures, or light are frequently reported signs — usually noticed between 12 and 36 months.",
  },
  {
    icon: TrendingUp,
    title: "Why early screening matters",
    description:
      "Research consistently links earlier identification with earlier access to support — and earlier support is associated with better developmental outcomes. Screening is a first step, not a verdict: it flags whether a fuller look is worth pursuing.",
  },
  {
    icon: Stethoscope,
    title: "Screening isn't diagnosis",
    description:
      "Tools like the AQ-10 and Q-CHAT-10 used here are validated screening instruments, not diagnostic tests. A formal diagnosis requires evaluation by a pediatrician, psychologist, or developmental specialist trained in autism assessment.",
  },
];

const PILLARS = [
  {
    icon: ClipboardList,
    title: "Behavioral screening items",
    description:
      "Ten Q-CHAT-10-style yes/no items, age-routed to a dedicated toddler or child model rather than a one-size-fits-all score.",
    step: "01",
  },
  {
    icon: Video,
    title: "Video observation",
    description:
      "Optional face/gaze/blink signal, plus an experimental (low-accuracy) ASD prediction model — both kept separate from the main screening result.",
    step: "02",
  },
  {
    icon: LayoutDashboard,
    title: "Individual characteristics",
    description:
      "Demographic and background context — age, family history, prior app use — that the model was trained to condition on.",
    step: "03",
  },
];

const TRUST_POINTS = [
  { icon: ShieldCheck, text: "Evidence-grounded explanations, citation-checked against source literature" },
  { icon: Sparkle, text: "Out-of-distribution and age-range flags instead of silent over-confident guesses" },
  { icon: History, text: "Every screening logged for audit, drift, and fairness monitoring" },
];

export function LandingPage() {
  return (
    <div>
      <section className="relative overflow-hidden">
        <div className="pointer-events-none absolute inset-0 bg-grid bg-radial-fade opacity-60" />
        <div className="mx-auto grid w-full max-w-7xl grid-cols-1 items-center gap-10 px-4 py-8 sm:px-6 md:py-12 lg:grid-cols-2 lg:px-8 lg:py-14">
          <div className="relative z-10 animate-fade-in">
            <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-border/70 bg-card/60 px-3 py-1 text-xs font-medium text-muted-foreground">
              <span className="h-1.5 w-1.5 rounded-full bg-success" />
              Research / educational prototype
            </div>
            <h1 className="text-4xl font-semibold leading-[1.1] tracking-tight sm:text-5xl lg:text-[3.25rem]">
              Autism screening support, <span className="text-gradient-brand">grounded in evidence</span>, not vibes.
            </h1>
            <p className="mt-4 max-w-xl text-base leading-relaxed text-muted-foreground sm:text-lg">
              NEUROSCREEN AI combines a behavioral questionnaire, optional video observation, and individual
              characteristics into one age-aware screening flow — with every explanation checked against cited
              sources before it reaches you.
            </p>
            <div className="mt-6 flex flex-col gap-3 sm:flex-row">
              <Button asChild size="lg" className="gap-2">
                <Link to="/screening">
                  Start a screening
                  <ArrowRight className="h-4 w-4" />
                </Link>
              </Button>
              <Button asChild variant="outline" size="lg">
                <Link to="/monitoring">View live monitoring</Link>
              </Button>
            </div>
            <p className="mt-6 text-xs leading-relaxed text-muted-foreground">
              This tool does not provide a diagnosis. Results should always be interpreted by a qualified
              professional.
            </p>
          </div>

          <div className="relative z-10 flex items-center justify-center">
            <div className="relative w-full max-w-lg animate-float">
              <div className="absolute inset-0 -z-10 rounded-full bg-primary/20 blur-3xl" />
              <Card className="card-elevated overflow-hidden">
                <CardContent className="p-0">
                  <NeuralHero className="h-64 w-full sm:h-80" />
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </section>

      <section className="border-t border-border/60 px-4 py-16 sm:px-6 md:py-20 lg:px-8">
        <div className="mx-auto w-full max-w-7xl">
          <div className="mx-auto max-w-2xl text-center">
            <h2 className="text-2xl font-semibold tracking-tight sm:text-3xl">Understanding autism spectrum disorder</h2>
            <p className="mt-3 text-sm text-muted-foreground sm:text-base">
              A quick primer before you screen — general background, not medical advice.
            </p>
          </div>

          <div className="mt-12 grid grid-cols-1 gap-5 sm:grid-cols-2">
            {ABOUT_AUTISM.map(({ icon: Icon, title, description }) => (
              <Card key={title} className="card-elevated">
                <CardContent className="flex gap-4 p-6">
                  <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border border-primary/25 bg-primary/10 text-primary">
                    <Icon className="h-5 w-5" />
                  </div>
                  <div>
                    <h3 className="text-base font-semibold">{title}</h3>
                    <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{description}</p>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          <p className="mx-auto mt-6 max-w-2xl text-center text-xs leading-relaxed text-muted-foreground">
            Every child develops differently, and having one or two of these traits doesn't mean a child is autistic.
            If you have concerns, a conversation with a pediatrician is always the right next step.
          </p>
        </div>
      </section>

      <section className="border-t border-border/60 bg-card/30 px-4 py-16 sm:px-6 md:py-20 lg:px-8">
        <div className="mx-auto w-full max-w-7xl">
          <div className="mx-auto max-w-2xl text-center">
            <h2 className="text-2xl font-semibold tracking-tight sm:text-3xl">Three inputs, one screening flow</h2>
            <p className="mt-3 text-sm text-muted-foreground sm:text-base">
              Each step is optional to explore and always separated from the model's decision boundary —
              the video panel's face/gaze metrics and its experimental prediction model are never
              fused into the main screening result.
            </p>
          </div>

          <div className="mt-12 grid grid-cols-1 gap-5 md:grid-cols-3">
            {PILLARS.map(({ icon: Icon, title, description, step }) => (
              <Card key={title} className="card-elevated relative overflow-hidden">
                <CardContent className="p-6">
                  <span className="text-xs font-mono text-muted-foreground/60">{step}</span>
                  <div className="mt-3 flex h-10 w-10 items-center justify-center rounded-lg border border-primary/25 bg-primary/10 text-primary">
                    <Icon className="h-5 w-5" />
                  </div>
                  <h3 className="mt-4 text-base font-semibold">{title}</h3>
                  <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{description}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      <section className="px-4 py-16 sm:px-6 md:py-20 lg:px-8">
        <div className="mx-auto grid w-full max-w-7xl grid-cols-1 gap-8 md:grid-cols-[1fr_1.2fr] md:items-center">
          <div>
            <h2 className="text-2xl font-semibold tracking-tight sm:text-3xl">Built for scrutiny, not just a score</h2>
            <p className="mt-3 text-sm leading-relaxed text-muted-foreground sm:text-base">
              Every prediction ships with the reasoning behind it — reviewable, checked, and monitored over time.
            </p>
          </div>
          <div className="space-y-3">
            {TRUST_POINTS.map(({ icon: Icon, text }) => (
              <div
                key={text}
                className="flex items-center gap-4 rounded-lg border border-border/60 bg-card/50 px-4 py-3.5"
              >
                <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-accent/60 text-accent-foreground">
                  <Icon className="h-[18px] w-[18px]" />
                </div>
                <p className="text-sm text-foreground/90">{text}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="border-t border-border/60 px-4 py-16 sm:px-6 lg:px-8">
        <div className="mx-auto flex w-full max-w-7xl flex-col items-center justify-between gap-6 rounded-2xl border border-primary/20 bg-gradient-to-br from-primary/10 via-card to-card px-6 py-10 text-center sm:flex-row sm:text-left">
          <div>
            <h2 className="text-xl font-semibold tracking-tight sm:text-2xl">Ready to run a screening?</h2>
            <p className="mt-2 text-sm text-muted-foreground">Takes about five minutes, end to end.</p>
          </div>
          <Button asChild size="lg" className="gap-2 whitespace-nowrap">
            <Link to="/screening">
              Start a screening
              <ArrowRight className="h-4 w-4" />
            </Link>
          </Button>
        </div>
      </section>
    </div>
  );
}
