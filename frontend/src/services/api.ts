import axios, { AxiosError } from "axios";
import type {
  AssessmentRequest,
  AssessmentResponse,
  AuditTrail,
  DriftSummary,
  FairnessSummary,
  HealthStatus,
  PerformanceSummary,
  PredictionResult,
  ScreeningDetail,
  ScreeningHistoryPage,
  VideoAsdPredictionResult,
  VideoObservationResult,
} from "@/types/api";

const client = axios.create({
  baseURL: "/",
  timeout: 120_000,
});

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

function unwrap<T>(promise: Promise<{ data: T }>): Promise<T> {
  return promise
    .then((res) => res.data)
    .catch((err: AxiosError<{ detail?: unknown }>) => {
      const status = err.response?.status ?? 0;
      const detail = err.response?.data?.detail;
      const message =
        typeof detail === "string"
          ? detail
          : Array.isArray(detail)
            ? detail.map((d) => JSON.stringify(d)).join("; ")
            : err.message;
      throw new ApiError(message, status);
    });
}

export const api = {
  health: () => unwrap<HealthStatus>(client.get("/health")),

  createAssessment: (payload: AssessmentRequest) =>
    unwrap<AssessmentResponse>(client.post("/api/v1/assessment", payload)),

  predict: (payload: AssessmentRequest) =>
    unwrap<PredictionResult>(client.post("/api/v1/predict", payload)),

  listHistory: (params: { limit?: number; offset?: number; route?: string } = {}) =>
    unwrap<ScreeningHistoryPage>(client.get("/api/v1/history", { params })),

  getHistoryItem: (id: string) => unwrap<ScreeningDetail>(client.get(`/api/v1/history/${id}`)),

  analyzeVideo: (file: Blob, filename: string) => {
    const form = new FormData();
    form.append("file", file, filename);
    return unwrap<VideoObservationResult>(
      client.post("/api/v1/video/analyze", form, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 180_000,
      }),
    );
  },

  predictVideoAsd: (file: Blob, filename: string) => {
    const form = new FormData();
    form.append("file", file, filename);
    return unwrap<VideoAsdPredictionResult>(
      client.post("/api/v1/video/predict-asd", form, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 180_000,
      }),
    );
  },

  monitoring: {
    performance: () => unwrap<PerformanceSummary>(client.get("/api/v1/monitoring/performance")),
    fairness: () => unwrap<FairnessSummary>(client.get("/api/v1/monitoring/fairness")),
    drift: () => unwrap<DriftSummary>(client.get("/api/v1/monitoring/drift")),
    audit: () => unwrap<AuditTrail>(client.get("/api/v1/monitoring/audit")),
  },
};
