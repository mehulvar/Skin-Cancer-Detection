import { describe, expect, it } from "vitest";
import { CLASSES, classByKey } from "../classes";
import { averageProbabilities, CONCERN_THRESHOLD, interpret, softmax } from "../interpret";
import { centerSquareCrop, flipHorizontal, rgbaToTensor } from "../preprocess";

const oneHot = (key: string, p = 0.9): number[] => {
  const rest = (1 - p) / (CLASSES.length - 1);
  return CLASSES.map((c) => (c.key === key ? p : rest));
};

describe("classes", () => {
  it("matches the model output order used in training", () => {
    expect(CLASSES.map((c) => c.key)).toEqual([
      "akiec", "bcc", "bkl", "df", "mel", "nv", "vasc", "other_skin", "not_skin",
    ]);
  });

  it("throws for unknown keys", () => {
    expect(() => classByKey("nope")).toThrow("Unknown class key");
  });
});

describe("softmax", () => {
  it("sums to one and preserves order", () => {
    const p = softmax([1, 2, 3]);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
    expect(p[2]).toBeGreaterThan(p[1]);
  });

  it("is numerically stable for large logits", () => {
    const p = softmax([1000, 1000]);
    expect(p[0]).toBeCloseTo(0.5);
  });

  it("rejects empty input", () => {
    expect(() => softmax([])).toThrow();
  });
});

describe("averageProbabilities", () => {
  it("averages element-wise", () => {
    const [a, b] = averageProbabilities([[0.2, 0.8], [0.4, 0.6]]);
    expect(a).toBeCloseTo(0.3);
    expect(b).toBeCloseTo(0.7);
  });

  it("rejects zero runs", () => {
    expect(() => averageProbabilities([])).toThrow();
  });
});

describe("interpret", () => {
  it("flags melanoma as concern", () => {
    const v = interpret(oneHot("mel"), CLASSES);
    expect(v.kind).toBe("concern");
    expect(v.top.info.key).toBe("mel");
    expect(v.lowConfidence).toBe(false);
  });

  it("calls a confident mole benign", () => {
    expect(interpret(oneHot("nv", 0.95), CLASSES).kind).toBe("benign");
  });

  it("escalates a benign top class when malignant probability is meaningful", () => {
    const special: Record<string, number> = { nv: 0.55, mel: CONCERN_THRESHOLD, bcc: 0.05 };
    const probs = CLASSES.map((c) => special[c.key] ?? 0.1 / 6);
    const v = interpret(probs, CLASSES);
    expect(v.top.info.key).toBe("nv");
    expect(v.kind).toBe("concern");
  });

  it("recognises non-skin and other skin conditions", () => {
    expect(interpret(oneHot("not_skin"), CLASSES).kind).toBe("not-skin");
    expect(interpret(oneHot("other_skin"), CLASSES).kind).toBe("other-skin");
  });

  it("marks low confidence and ranks descending", () => {
    const v = interpret(oneHot("bkl", 0.4), CLASSES);
    expect(v.lowConfidence).toBe(true);
    const ps = v.ranked.map((r) => r.probability);
    expect([...ps].sort((a, b) => b - a)).toEqual(ps);
  });

  it("rejects a score vector of the wrong length", () => {
    expect(() => interpret([0.5, 0.5], CLASSES)).toThrow("scores");
  });
});

describe("preprocess", () => {
  it("centre-crops landscape and portrait images", () => {
    expect(centerSquareCrop(400, 200)).toEqual({ sx: 100, sy: 0, side: 200 });
    expect(centerSquareCrop(100, 301)).toEqual({ sx: 0, sy: 100, side: 100 });
    expect(() => centerSquareCrop(0, 10)).toThrow();
  });

  it("normalises RGBA into CHW planes", () => {
    const rgba = new Uint8ClampedArray([255, 0, 51, 255, 0, 255, 102, 255, 0, 0, 0, 255, 255, 255, 255, 255]);
    const t = rgbaToTensor(rgba, { size: 2, mean: [0, 0, 0], std: [1, 1, 1] });
    expect(Array.from(t.slice(0, 4))).toEqual([1, 0, 0, 1]); // R plane
    expect(Array.from(t.slice(4, 8))).toEqual([0, 1, 0, 1]); // G plane
    expect(t[8]).toBeCloseTo(0.2);
  });

  it("applies mean/std", () => {
    const rgba = new Uint8ClampedArray([255, 255, 255, 255]);
    const t = rgbaToTensor(rgba, { size: 1, mean: [0.5, 0.5, 0.5], std: [0.5, 0.25, 0.5] });
    expect(Array.from(t)).toEqual([1, 2, 1]);
  });

  it("rejects mismatched buffers", () => {
    expect(() => rgbaToTensor(new Uint8ClampedArray(3), { size: 1, mean: [0, 0, 0], std: [1, 1, 1] })).toThrow();
  });

  it("flips each row of each channel", () => {
    const t = Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8]); // 2 channels of 2x2
    expect(Array.from(flipHorizontal(t, 2))).toEqual([2, 1, 4, 3, 6, 5, 8, 7]);
  });
});
