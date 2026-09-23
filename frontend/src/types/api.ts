// Mirrors backend/app/schemas/*.py. Keep in sync by hand — this is a
// small enough surface that a generated client would be more ceremony
// than value right now.

export type PredictedLabel = "YES" | "NO";
export type UncertaintyCategory = "LOW" | "MEDIUM" | "HIGH";
export type DistributionStatus = "IN_DISTRIBUTION" | "OUT_OF_DISTRIBUTION" | "OOD_NOT_AVAILABLE";
export type DecisionStatus = "SCREENING_RESULT" | "HUMAN_REVIEW" | "OOD_REVIEW";
export type RouteType = "STANDARD_REVIEW" | "HUMAN_REVIEW" | "OOD_REVIEW" | "ERROR";
export type SafetyStatus = "SCREENING_ONLY" | "BLOCKED";

export interface AssessmentRequest {
  A1_Score: number | null;
  A2_Score: number | null;
  A3_Score: number | null;
  A4_Score: number | null;
  A5_Score: number | null;
  A6_Score: number | null;
  A7_Score: number | null;
  A8_Score: number | null;
  A9_Score: number | null;
  A10_Score: number | null;
  age: number | null;
  gender: string | null;
  ethnicity: string | null;
  jundice: string | null;
  austim: string | null;
  contry_of_res: string | null;
  used_app_before: string | null;
  relation: string | null;
}

export interface Prediction {
  label: PredictedLabel;
  probability_yes: number;
  probability_no: number;
}

export interface Uncertainty {
  confidence: number;
  uncertainty: number;
  category: UncertaintyCategory;
}

export interface Distribution {
  status: DistributionStatus;
  distance: number | null;
  threshold: number | null;
  is_ood: boolean;
  age_out_of_range: boolean;
}

export interface Decision {
  status: DecisionStatus;
  reason: string;
  uncertainty_category: UncertaintyCategory;
  confidence: number;
}

export interface EvidenceItem {
  chunk_id?: string | null;
  citation_id?: string | null;
  source?: string | null;
  file_name?: string | null;
  file_type?: string | null;
  chunk_index?: number | null;
  text: string;
  similarity?: number | null;
  rerank_score?: number | null;
  rank?: number | null;
}

export interface Citation {
  citation_id: string;
  source?: string | null;
  file_name?: string | null;
  chunk_id?: string | null;
  similarity?: number | null;
  text: string;
}

export interface ClaimEvaluation {
  claim: string;
  citation_ids: string[];
  supported: boolean;
  support_score: number;
  matched_evidence: string[];
}

export interface GroundednessThresholds {
  groundedness_threshold: number;
  claim_support_threshold: number;
  max_unsupported_claim_rate: number;
}

export interface GroundednessResult {
  groundedness_score: number;
  citation_coverage: number;
  citation_precision: number;
  unsupported_claim_rate: number;
  total_claims: number;
  supported_claims: number;
  unsupported_claims: number;
  cited_claims: number;
  claims: ClaimEvaluation[];
  passed: boolean;
  thresholds: GroundednessThresholds;
  error?: string | null;
}

export interface AuditChecks {
  non_empty: boolean;
  diagnostic_claims: boolean;
  probability_claims: boolean;
  reliability_claims: boolean;
  causal_claims: boolean;
  feature_claims: boolean;
  individual_interpretation: boolean;
  citations: boolean;
  groundedness: boolean;
  unsupported_claims: boolean;
}

export interface AuditResult {
  llm_status: string;
  final_status: string;
  fallback_used: boolean;
  reason: string;
  checks: AuditChecks;
  matched_pattern?: string | null;
  groundedness: {
    score: number;
    citation_coverage: number;
    citation_precision: number;
    unsupported_claim_rate: number;
    total_claims: number;
    supported_claims: number;
    unsupported_claims: number;
  };
}

export interface ModelInfo {
  name: string;
  version: string;
  status: string;
  age_band: string;
}

export interface AssessmentResponse {
  summary: string;
  prediction: Prediction;
  model: ModelInfo;
  uncertainty: Uncertainty;
  distribution: Distribution;
  decision: Decision;
  route: RouteType;
  route_reason: string;
  evidence: EvidenceItem[];
  citations: Citation[];
  safety_status: SafetyStatus;
  safety_notes: string[];
  llm_explanation: string;
  llm_explanation_original: string;
  generator: string;
  fallback_used: boolean;
  final_explanation_status: string;
  groundedness: GroundednessResult;
  groundedness_passed: boolean;
  audit: AuditResult;
  disclaimer: string;
}

export interface FeatureContribution {
  feature: string;
  shap_value: number;
  direction: string;
}

export interface Explanation {
  available: boolean;
  output_scale?: string | null;
  note?: string | null;
  top_contributions: FeatureContribution[];
}

export interface PredictionResult {
  model: ModelInfo;
  prediction: Prediction;
  uncertainty: Uncertainty;
  distribution: Distribution;
  decision: Decision;
  explanation: Explanation;
  metadata: {
    feature_count: number;
    feature_names: string[];
    excluded_features: string[];
    transformed_feature_count: number;
  };
  disclaimer: string;
}

export interface ActiveModelSummary {
  name: string;
  version: string;
  status: string;
  age_range_years: number[] | null;
}

export interface HealthStatus {
  status: string;
  environment: string;
  models: Record<string, ActiveModelSummary>;
  rag_index_available: boolean;
}

export interface ModelRegistryEntry {
  name: string;
  version: string;
  status: string;
  artifact: string;
  metadata: string;
  age_range_years: number[] | null;
}

export interface ModelRegistryInfo {
  active_models: Record<string, ModelRegistryEntry>;
  updated_at: string | null;
}

export interface ScreeningSummary {
  id: string;
  created_at: string;
  model_version: string;
  prediction_label: PredictedLabel;
  probability_yes: number;
  uncertainty_category: UncertaintyCategory;
  distribution_status: DistributionStatus;
  decision_status: DecisionStatus;
  route: RouteType;
  safety_status: SafetyStatus;
  groundedness_score: number;
  groundedness_passed: boolean;
  fallback_used: boolean;
  audit_status: string;
}

export interface ScreeningDetail extends ScreeningSummary {
  patient_data: Record<string, unknown>;
  full_response: AssessmentResponse;
}

export interface ScreeningHistoryPage {
  total: number;
  limit: number;
  offset: number;
  items: ScreeningSummary[];
}

export interface PerformanceSummary {
  total_screenings: number;
  label_distribution: Record<string, number>;
  route_distribution: Record<string, number>;
  decision_distribution: Record<string, number>;
  avg_probability_yes: number | null;
  avg_groundedness_score: number | null;
  groundedness_pass_rate: number | null;
  fallback_rate: number | null;
  audit_pass_rate: number | null;
}

export interface SubgroupStats {
  n: number;
  positive_rate: number | null;
  avg_groundedness_score: number | null;
}

export interface FairnessSummary {
  disclaimer: string;
  total_screenings: number;
  breakdown: Record<string, Record<string, SubgroupStats>>;
}

export interface VideoObservationResult {
  frames_analyzed: number;
  frames_with_face_detected: number;
  face_detected_ratio: number;
  avg_gaze_toward_camera_ratio: number | null;
  head_pose_variability: number | null;
  blink_rate_per_minute: number | null;
  duration_seconds: number;
  disclaimer: string;
}

export interface VideoAsdPredictionResult {
  predicted_label: "ASD" | "TD";
  probability_asd: number;
  probability_td: number;
  segments_analyzed: number;
  frames_analyzed: number;
  model_version: string;
  test_accuracy: number;
  test_f1: number;
  disclaimer: string;
}

export interface DriftSummary {
  total_screenings: number;
  recent_window: number;
  ood_rate_recent: number | null;
  ood_rate_overall: number | null;
  avg_probability_recent: number | null;
  avg_probability_overall: number | null;
  ood_drift_detected: boolean;
  prediction_drift_detected: boolean;
}

export interface AuditEntry {
  id: string;
  timestamp: string;
  model_version: string;
  route: string;
  safety_status: string;
  prediction_label: string;
  probability_yes: number;
  distribution_status: string;
  decision_status: string;
  groundedness_score: number;
  groundedness_passed: boolean;
  fallback_used: boolean;
  audit_status: string;
}

export interface AuditTrail {
  total: number;
  entries: AuditEntry[];
}

export interface ApiErrorBody {
  error: string;
  detail: string | Record<string, unknown>[];
  status_code: number;
}
