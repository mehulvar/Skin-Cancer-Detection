import type { Category, ClassInfo } from "./classes";

/** Combined malignant + precancerous probability above which we always advise a check. */
export const CONCERN_THRESHOLD = 0.3;
/** Top-class probability below which the result is labelled low-confidence. */
export const LOW_CONFIDENCE_THRESHOLD = 0.5;

export interface RankedClass {
  info: ClassInfo;
  probability: number;
}

export type VerdictKind = "concern" | "benign" | "other-skin" | "not-skin";

export interface Verdict {
  kind: VerdictKind;
  top: RankedClass;
  ranked: RankedClass[];
  concernProbability: number;
  lowConfidence: boolean;
}

const CONCERN_CATEGORIES: ReadonlySet<Category> = new Set(["malignant", "precancerous"]);

export function softmax(logits: ArrayLike<number>): number[] {
  const values = Array.from(logits);
  if (values.length === 0) {
    throw new Error("softmax of empty logits");
  }
  const max = Math.max(...values);
  const exps = values.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

export function averageProbabilities(runs: number[][]): number[] {
  if (runs.length === 0) {
    throw new Error("no probability runs to average");
  }
  return runs[0].map((_, i) => runs.reduce((acc, r) => acc + r[i], 0) / runs.length);
}

export function interpret(probabilities: number[], classes: readonly ClassInfo[]): Verdict {
  if (probabilities.length !== classes.length) {
    throw new Error(`Model returned ${probabilities.length} scores for ${classes.length} classes`);
  }
  const ranked = classes
    .map((info, i) => ({ info, probability: probabilities[i] }))
    .sort((a, b) => b.probability - a.probability);
  const top = ranked[0];
  const concernProbability = ranked
    .filter((r) => CONCERN_CATEGORIES.has(r.info.category))
    .reduce((acc, r) => acc + r.probability, 0);

  return {
    kind: verdictKind(top.info.category, concernProbability),
    top,
    ranked,
    concernProbability,
    lowConfidence: top.probability < LOW_CONFIDENCE_THRESHOLD,
  };
}

function verdictKind(topCategory: Category, concernProbability: number): VerdictKind {
  if (topCategory === "invalid") return "not-skin";
  if (CONCERN_CATEGORIES.has(topCategory) || concernProbability >= CONCERN_THRESHOLD) return "concern";
  if (topCategory === "non-lesion") return "other-skin";
  return "benign";
}
