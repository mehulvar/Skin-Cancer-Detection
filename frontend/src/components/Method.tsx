import { METRICS } from "@/lib/content";

const STEPS = [
  {
    letter: "D",
    title: "Data",
    body: "HAM10000 / ISIC 2018 (10,015 dermoscopic images, 7 classes), 4,500 photos of non-cancerous conditions from Google's SCIN dataset, and 6,000 everyday images and textures from Imagenette and DTD.",
  },
  {
    letter: "S",
    title: "Leakage-free splits",
    body: "HAM10000 often has several photos of the same lesion. Splits are grouped by lesion ID so no lesion appears in both training and validation, and the final score uses the separate official ISIC test set.",
  },
  {
    letter: "T",
    title: "Transfer learning",
    body: "An ImageNet-pretrained ConvNeXt-Nano is fine-tuned with heavy augmentation (rotations, flips, colour jitter, random crops), label smoothing, square-root class-balanced sampling and a cosine learning-rate schedule.",
  },
  {
    letter: "B",
    title: "In-browser inference",
    body: "The trained network is exported to ONNX and executed by ONNX Runtime Web. Each image is scored twice (original + mirror) and averaged for a steadier prediction.",
  },
];

export function Method() {
  return (
    <section id="method" className="py-20 md:py-28">
      <div className="mx-auto grid max-w-6xl gap-12 px-4 md:grid-cols-[1fr_1.4fr] md:px-8">
        <div>
          <p className="eyebrow">Method</p>
          <h2 className="mt-3 font-serif text-4xl leading-tight text-ink md:text-6xl">How it was built.</h2>
          <p className="mt-5 text-muted">
            SkinSight started as a research project with a hand-built CNN. This version keeps the same goal and
            dataset but rebuilds the pipeline around modern, verifiable practice.
          </p>
          {METRICS && (
            <p className="mt-5 text-sm text-muted">
              Model: <span className="text-text">{METRICS.architecture}</span> · {METRICS.modelMb.toFixed(0)} MB download
            </p>
          )}
        </div>
        <ul className="flex flex-col divide-y divide-line border-y border-line">
          {STEPS.map((s) => (
            <li key={s.title} className="grid grid-cols-[3rem_1fr] gap-5 py-7">
              <span className="flex h-11 w-11 items-center justify-center rounded-full border border-clay font-serif text-xl text-clay-deep">
                {s.letter}
              </span>
              <div>
                <h3 className="text-lg font-medium text-ink">{s.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-muted">{s.body}</p>
              </div>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
