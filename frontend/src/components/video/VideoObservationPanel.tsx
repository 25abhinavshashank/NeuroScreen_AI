import { useCallback, useEffect, useRef, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { AlertTriangle, Camera, Circle, FlaskConical, Info, Loader2, Square, Upload } from "lucide-react";
import { toast } from "sonner";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { formatPercent } from "@/lib/utils";
import { api, ApiError } from "@/services/api";
import type { VideoAsdPredictionResult, VideoObservationResult } from "@/types/api";

type Mode = "upload" | "webcam";

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border bg-card p-3">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="mt-1 text-xl font-semibold tabular-nums">{value}</div>
    </div>
  );
}

export function VideoObservationPanel() {
  const [mode, setMode] = useState<Mode>("upload");
  const [result, setResult] = useState<VideoObservationResult | null>(null);
  const [asdResult, setAsdResult] = useState<VideoAsdPredictionResult | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordSeconds, setRecordSeconds] = useState(0);
  const [hasStream, setHasStream] = useState(false);
  const [hasFile, setHasFile] = useState(false);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const lastFileRef = useRef<{ file: Blob; filename: string } | null>(null);

  const mutation = useMutation({
    mutationFn: ({ file, filename }: { file: Blob; filename: string }) => api.analyzeVideo(file, filename),
    onSuccess: (data) => {
      setResult(data);
      toast.success("Video analysis complete");
    },
    onError: (err: unknown) => {
      const message = err instanceof ApiError ? err.message : "Video analysis failed.";
      toast.error("Analysis failed", { description: message });
    },
  });

  const asdMutation = useMutation({
    mutationFn: ({ file, filename }: { file: Blob; filename: string }) => api.predictVideoAsd(file, filename),
    onSuccess: (data) => {
      setAsdResult(data);
    },
    onError: (err: unknown) => {
      const message = err instanceof ApiError ? err.message : "Experimental prediction failed.";
      toast.error("Prediction failed", { description: message });
    },
  });

  function runAsdPrediction() {
    if (!lastFileRef.current) return;
    setAsdResult(null);
    asdMutation.mutate(lastFileRef.current);
  }

  const stopStream = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setHasStream(false);
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  useEffect(() => stopStream, [stopStream]);

  async function startWebcam() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      streamRef.current = stream;
      setHasStream(true);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
    } catch {
      toast.error("Couldn't access the webcam", {
        description: "Check that camera permission is granted for this site.",
      });
    }
  }

  function startRecording() {
    if (!streamRef.current) return;
    chunksRef.current = [];
    const recorder = new MediaRecorder(streamRef.current, { mimeType: "video/webm" });
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };
    recorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: "video/webm" });
      const filename = "webcam-recording.webm";
      lastFileRef.current = { file: blob, filename };
      setHasFile(true);
      setAsdResult(null);
      mutation.mutate({ file: blob, filename });
    };
    recorder.start();
    recorderRef.current = recorder;
    setIsRecording(true);
    setRecordSeconds(0);
    timerRef.current = setInterval(() => setRecordSeconds((s) => s + 1), 1000);
  }

  function stopRecording() {
    recorderRef.current?.stop();
    setIsRecording(false);
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    lastFileRef.current = { file, filename: file.name };
    setHasFile(true);
    setResult(null);
    setAsdResult(null);
    mutation.mutate({ file, filename: file.name });
  }

  function switchMode(next: Mode) {
    stopStream();
    setIsRecording(false);
    setResult(null);
    setAsdResult(null);
    lastFileRef.current = null;
    setHasFile(false);
    setMode(next);
    if (next === "webcam") {
      void startWebcam();
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Video observation (experimental)</CardTitle>
        <CardDescription>
          Optional facial/gaze tracking from a video file or webcam recording, plus an
          experimental ASD prediction model trained on real clinical video data. Neither is
          combined with the screening result above — see the disclaimers below for what each
          one actually tells you.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <Alert>
          <Info className="h-4 w-4" />
          <AlertDescription>
            The face/gaze/blink metrics below have no validated connection to ASD outcomes — they're
            raw descriptive observations only. A separate model further down was trained on real
            labeled video data and does attempt a prediction, but its accuracy is barely above chance
            — see its own warning before drawing any conclusion from it.
          </AlertDescription>
        </Alert>

        <div className="flex gap-2">
          <Button
            type="button"
            variant={mode === "upload" ? "default" : "outline"}
            size="sm"
            onClick={() => switchMode("upload")}
            className="gap-1.5"
          >
            <Upload className="h-3.5 w-3.5" />
            Upload video
          </Button>
          <Button
            type="button"
            variant={mode === "webcam" ? "default" : "outline"}
            size="sm"
            onClick={() => switchMode("webcam")}
            className="gap-1.5"
          >
            <Camera className="h-3.5 w-3.5" />
            Use webcam
          </Button>
        </div>

        {mode === "upload" && (
          <div className="space-y-2">
            <input
              type="file"
              accept="video/mp4,video/webm,video/quicktime,video/x-matroska"
              onChange={handleFileChange}
              disabled={mutation.isPending}
              className="block w-full text-sm text-muted-foreground file:mr-3 file:rounded-md file:border-0 file:bg-secondary file:px-3 file:py-1.5 file:text-sm file:font-medium hover:file:bg-secondary/80"
            />
          </div>
        )}

        {mode === "webcam" && (
          <div className="space-y-3">
            <video
              ref={videoRef}
              muted
              playsInline
              className="aspect-video w-full max-w-md rounded-lg border bg-black object-cover"
            />
            <div className="flex items-center gap-3">
              {!isRecording ? (
                <Button
                  type="button"
                  size="sm"
                  onClick={startRecording}
                  disabled={!hasStream || mutation.isPending}
                  className="gap-1.5"
                >
                  <Circle className="h-3.5 w-3.5 fill-current" />
                  Start recording
                </Button>
              ) : (
                <Button type="button" size="sm" variant="destructive" onClick={stopRecording} className="gap-1.5">
                  <Square className="h-3.5 w-3.5 fill-current" />
                  Stop ({recordSeconds}s)
                </Button>
              )}
            </div>
          </div>
        )}

        {mutation.isPending && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            Analyzing video…
          </div>
        )}

        {result && (
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
              <Metric label="Face detected" value={formatPercent(result.face_detected_ratio)} />
              <Metric
                label="Gaze toward camera"
                value={result.avg_gaze_toward_camera_ratio !== null ? formatPercent(result.avg_gaze_toward_camera_ratio) : "—"}
              />
              <Metric
                label="Head pose variability"
                value={result.head_pose_variability !== null ? result.head_pose_variability.toFixed(1) : "—"}
              />
              <Metric
                label="Blink rate"
                value={result.blink_rate_per_minute !== null ? `${result.blink_rate_per_minute.toFixed(1)}/min` : "—"}
              />
              <Metric label="Duration" value={`${result.duration_seconds.toFixed(1)}s`} />
              <Metric label="Frames analyzed" value={String(result.frames_analyzed)} />
            </div>
            <Progress value={result.face_detected_ratio * 100} className="h-1.5" />
            <Alert>
              <Info className="h-4 w-4" />
              <AlertDescription className="text-xs">{result.disclaimer}</AlertDescription>
            </Alert>
          </div>
        )}

        {hasFile && (
          <div className="space-y-3 rounded-lg border border-warning/30 bg-warning/5 p-4">
            <div className="flex items-center gap-2">
              <FlaskConical className="h-4 w-4 text-warning" />
              <span className="text-sm font-semibold">Experimental ASD prediction</span>
              <Badge variant="warning">59% test accuracy</Badge>
            </div>
            <p className="text-xs leading-relaxed text-muted-foreground">
              A research classifier trained on real (but limited) labeled clinical video data. Its
              measured accuracy on held-out subjects is barely above a coin flip — this is shown for
              transparency, not because it's trustworthy. It is never combined with the screening
              result above.
            </p>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={runAsdPrediction}
              disabled={asdMutation.isPending}
              className="gap-1.5 border-warning/40 text-warning hover:bg-warning/10"
            >
              {asdMutation.isPending ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <FlaskConical className="h-3.5 w-3.5" />
              )}
              Run experimental prediction
            </Button>

            {asdResult && (
              <div className="space-y-3 border-t border-warning/20 pt-3">
                <div className="flex flex-wrap items-center gap-3">
                  <Badge variant={asdResult.predicted_label === "ASD" ? "warning" : "success"} className="text-sm px-3 py-1">
                    Predicted: {asdResult.predicted_label}
                  </Badge>
                  <span className="text-sm text-muted-foreground">
                    {formatPercent(asdResult.probability_asd)} ASD / {formatPercent(asdResult.probability_td)} TD
                  </span>
                  <span className="text-xs text-muted-foreground">
                    ({asdResult.segments_analyzed} segment{asdResult.segments_analyzed === 1 ? "" : "s"} analyzed)
                  </span>
                </div>
                <Alert variant="warning">
                  <AlertTriangle className="h-4 w-4" />
                  <AlertDescription className="text-xs">{asdResult.disclaimer}</AlertDescription>
                </Alert>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
