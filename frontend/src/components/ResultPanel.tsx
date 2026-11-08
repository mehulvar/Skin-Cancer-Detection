import { CATEGORY_LABEL, classByKey, type Category } from "@/lib/classes";
import { percent } from "@/lib/content";
import type { Prediction } from "@/lib/inference";
import type { VerdictKind } from "@/lib/interpret";

const SEVERITY_BG: Record<Category, string> = {
  malignant: "bg-sev-malignant",
  precancerous: "bg-sev-precancerous",
  benign: "bg-sev-benign",
  "non-lesion": "bg-sev-nonlesion",
  invalid: "bg-sev-invalid",
};

const SEVERITY_TEXT: Record<Category, string> = {
  malignant: "text-sev-malignant",
  precancerous: "text-sev-precancerous",
  benign: "text-sev-benign",
  "non-lesion": "text-sev-nonlesion",
  invalid: "text-sev-invalid",
};

const GUIDANCE: Record<VerdictKind, { title: string; body: string }> = {
  concern: {
    title: "Worth showing a dermatologist",
    body: "The model sees features associated with skin cancer or a precancerous lesion. This is not a diagnosis — but it is a good reason to book a skin check soon.",
  },
  benign: {
    title: "Looks benign to the model",
    body: "The model's strongest match is a harmless lesion type. Keep an eye on it, and get any mark that changes, bleeds or itches checked in person.",
  },
  "other-skin": {
    title: "Looks like a non-cancerous skin condition",
    body: "This resembles a rash or inflammatory condition (eczema, dermatitis, hives…) rather than a tumour-type lesion. A GP or dermatologist can confirm and treat it.",
  },
  "not-skin": {
    title: "This doesn't look like skin",
    body: "The model thinks this isn't a photo of skin, so it won't attempt a lesion diagnosis. Try a close-up, well-lit image of a single skin lesion.",
  },
};

export function CategoryBadge({ category }: { category: Category }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-line bg-cream px-2.5 py-0.5 text-xs font-medium text-text">
      <span className={`h-2 w-2 rounded-full ${SEVERITY_BG[category]}`} aria-hidden />
      {CATEGORY_LABEL[category]}
    </span>
  );
}

interface ResultPanelProps {
  prediction: Prediction;
  trueLabel?: string;
}

export function ResultPanel({ prediction, trueLabel }: ResultPanelProps) {
  const { verdict, elapsedMs } = prediction;
  const { top } = verdict;
  const guidance = GUIDANCE[verdict.kind];
  const truth = trueLabel ? classByKey(trueLabel) : null;
  const correct = truth ? truth.key === top.info.key : null;

  return (
    <div className="rise flex flex-col gap-6" data-testid="result" data-prediction={top.info.key}>
      <div>
        <p className="eyebrow">Most likely</p>
        <h3 className={`mt-2 font-serif text-4xl leading-tight md:text-5xl ${SEVERITY_TEXT[top.info.category]}`}>
          {top.info.short}
        </h3>
        <div className="mt-3 flex flex-wrap items-center gap-2 text-sm text-muted">
          <CategoryBadge category={top.info.category} />
          <span>{percent(top.probability)} confidence</span>
          {verdict.lowConfidence && (
            <span className="rounded-full bg-linen px-2.5 py-0.5 text-xs text-cocoa">Low confidence</span>
          )}
        </div>
      </div>

      {truth && (
        <p className="rounded-2xl border border-line bg-sand px-4 py-3 text-sm">
          <span className="text-muted">Ground truth: </span>
          <span className="font-medium">{truth.short}</span>
          <span className={`ml-2 font-medium ${correct ? "text-sev-benign" : "text-sev-malignant"}`}>
            {correct ? "✓ correct" : "✗ missed"}
          </span>
        </p>
      )}

      <div className="rounded-2xl bg-ink p-5 text-cream">
        <p className="font-serif text-2xl">{guidance.title}</p>
        <p className="mt-2 text-sm leading-relaxed text-cream/80">{guidance.body}</p>
        {verdict.kind !== "not-skin" && (
          <p className="mt-3 text-xs text-cream/60">
            Combined malignant + precancerous probability: {percent(verdict.concernProbability)}
          </p>
        )}
      </div>

      <div>
        <p className="eyebrow mb-3">All classes</p>
        <ul className="flex flex-col gap-2.5">
          {verdict.ranked.map(({ info, probability }) => (
            <li key={info.key} className="grid grid-cols-[minmax(0,9.5rem)_1fr_3.5rem] items-center gap-3 text-sm">
              <span className="truncate" title={info.name}>{info.short}</span>
              <span className="h-2 overflow-hidden rounded-full bg-linen">
                <span
                  className={`block h-full rounded-full ${SEVERITY_BG[info.category]}`}
                  style={{ width: `${Math.max(probability * 100, 0.5)}%` }}
                />
              </span>
              <span className="text-right tabular-nums text-muted">{percent(probability)}</span>
            </li>
          ))}
        </ul>
      </div>

      <p className="text-xs text-muted">
        Analysed on your device in {Math.round(elapsedMs)} ms · educational use only, not medical advice.
      </p>
    </div>
  );
}
