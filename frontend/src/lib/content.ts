/** Typed access to the generated data files in src/data (written by training/skinlesion). */

import metaJson from "@/data/model-meta.json";
import samplesJson from "@/data/samples.json";
import { classByKey, type Category } from "./classes";

export interface Sample {
  id: string;
  src: string;
  label: string;
  source: string;
}

export interface HeadlineMetrics {
  testImages: number;
  accuracy: number;
  balancedAccuracy: number;
  hamBalancedAccuracy: number;
  hamAccuracy: number;
  hamTestImages: number;
  dangerSensitivity: number;
  dangerSpecificity: number;
  perClassRecall: Record<string, number>;
  modelMb: number;
  architecture: string;
}

export type SampleGroup = "cancer" | "benign" | "other" | "not-skin";

export const SAMPLE_GROUPS: readonly { key: SampleGroup; label: string }[] = [
  { key: "cancer", label: "Cancerous & precancerous" },
  { key: "benign", label: "Benign lesions" },
  { key: "other", label: "Other skin conditions" },
  { key: "not-skin", label: "Not skin" },
];

const GROUP_BY_CATEGORY: Record<Category, SampleGroup> = {
  malignant: "cancer",
  precancerous: "cancer",
  benign: "benign",
  "non-lesion": "other",
  invalid: "not-skin",
};

export function sampleGroup(sample: Sample): SampleGroup {
  return GROUP_BY_CATEGORY[classByKey(sample.label).category];
}

export const SAMPLES: readonly Sample[] = samplesJson as Sample[];
export const METRICS: HeadlineMetrics | null =
  (metaJson as { metrics: HeadlineMetrics | null }).metrics ?? null;

export function percent(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}
