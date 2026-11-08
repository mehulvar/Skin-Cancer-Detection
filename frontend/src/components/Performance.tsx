import { classByKey } from "@/lib/classes";
import { METRICS, MODEL_IMAGE_SIZE, percent } from "@/lib/content";
import baseline from "@/data/baseline.json";

interface BaselineMetrics {
  hamAccuracy: number | null;
  hamBalancedAccuracy: number | null;
  reportedValAccuracy: number;
}

const BASE = baseline as BaselineMetrics;

function Stat({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <div className="rounded-[1.5rem] border border-cream/15 p-6">
      <p className="text-xs uppercase tracking-[0.2em] text-cream/60">{label}</p>
      <p className="mt-3 font-serif text-5xl text-cream">{value}</p>
      <p className="mt-2 text-sm text-cream/70">{note}</p>
    </div>
  );
}

export function Performance() {
  if (!METRICS) return null;
  const m = METRICS;
  const recalls = Object.entries(m.perClassRecall);

  return (
    <section id="performance" className="bg-ink py-20 text-cream md:py-28">
      <div className="mx-auto max-w-6xl px-4 md:px-8">
        <p className="eyebrow !text-clay">Performance</p>
        <h2 className="mt-3 max-w-3xl font-serif text-4xl leading-tight md:text-6xl">
          Method, not myth — every number from held-out data.
        </h2>
        <p className="mt-4 max-w-2xl text-cream/70">
          Evaluated on {m.testImages.toLocaleString()} images the model never saw during training, including the{" "}
          {m.hamTestImages.toLocaleString()}-image official ISIC 2018 Task 3 test set.
        </p>

        <div className="mt-12 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Stat label="Overall accuracy" value={percent(m.accuracy)} note="All 9 classes, full test set" />
          <Stat label="Lesion accuracy" value={percent(m.hamAccuracy)} note="ISIC 2018 test set, 7 lesion types" />
          <Stat
            label="Balanced accuracy"
            value={percent(m.hamBalancedAccuracy)}
            note="Mean recall over the 7 lesion types — the official ISIC metric"
          />
          <Stat
            label="Cancer sensitivity"
            value={percent(m.dangerSensitivity)}
            note={`Malignant/precancerous flagged · ${percent(m.dangerSpecificity)} specificity`}
          />
        </div>

        <div className="mt-14 grid gap-10 lg:grid-cols-2">
          <div>
            <h3 className="font-serif text-3xl">Recall by class</h3>
            <p className="mt-2 text-sm text-cream/70">Share of each class the model identifies correctly.</p>
            <ul className="mt-6 flex flex-col gap-3">
              {recalls.map(([key, recall]) => (
                <li key={key} className="grid grid-cols-[10rem_1fr_3.5rem] items-center gap-3 text-sm">
                  <span className="truncate text-cream/85">{classByKey(key).short}</span>
                  <span className="h-2 overflow-hidden rounded-full bg-cream/10">
                    <span className="block h-full rounded-full bg-clay" style={{ width: `${recall * 100}%` }} />
                  </span>
                  <span className="text-right tabular-nums text-cream/70">{percent(recall, 0)}</span>
                </li>
              ))}
            </ul>

            <h3 className="mt-12 font-serif text-3xl">Then and now</h3>
            <div className="mt-4 overflow-x-auto">
              <table className="w-full min-w-[28rem] text-left text-sm">
                <thead className="text-cream/60">
                  <tr className="border-b border-cream/15">
                    <th className="py-2 pr-4 font-normal">ISIC 2018 test set</th>
                    <th className="py-2 pr-4 font-normal">Original CNN</th>
                    <th className="py-2 font-normal">SkinSight</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-cream/10">
                    <td className="py-2.5 pr-4 text-cream/80">Accuracy</td>
                    <td className="py-2.5 pr-4 tabular-nums">{BASE.hamAccuracy !== null ? percent(BASE.hamAccuracy) : "—"}</td>
                    <td className="py-2.5 tabular-nums">{percent(m.hamAccuracy)}</td>
                  </tr>
                  <tr className="border-b border-cream/10">
                    <td className="py-2.5 pr-4 text-cream/80">Balanced accuracy</td>
                    <td className="py-2.5 pr-4 tabular-nums">
                      {BASE.hamBalancedAccuracy !== null ? percent(BASE.hamBalancedAccuracy) : "—"}
                    </td>
                    <td className="py-2.5 tabular-nums">{percent(m.hamBalancedAccuracy)}</td>
                  </tr>
                  <tr>
                    <td className="py-2.5 pr-4 text-cream/80">Architecture</td>
                    <td className="py-2.5 pr-4">3-layer CNN, 128 px</td>
                    <td className="py-2.5">2× ConvNeXt-Nano ensemble, {MODEL_IMAGE_SIZE} px</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="mt-3 text-xs text-cream/55">
              The original CNN reported {percent(BASE.reportedValAccuracy, 0)} validation accuracy on a random split; it
              was re-trained here with the same architecture on the same lesion-grouped data for a fair comparison.
            </p>
          </div>

          <figure className="rounded-[1.75rem] bg-cream p-4">
            {/* eslint-disable-next-line @next/next/no-img-element -- static report image */}
            <img
              src="/model/confusion_matrix.png"
              alt="Row-normalised confusion matrix on the test set"
              loading="lazy"
              className="w-full rounded-xl"
            />
            <figcaption className="px-2 pt-3 text-xs text-muted">
              Confusion matrix on the test set (rows sum to 1). Most errors are between visually similar lesions —
              melanoma vs. mole, and benign keratosis vs. actinic keratosis.
            </figcaption>
          </figure>
        </div>
      </div>
    </section>
  );
}
