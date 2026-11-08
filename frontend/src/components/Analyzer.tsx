"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { SAMPLES, SAMPLE_GROUPS, sampleGroup, type Sample, type SampleGroup } from "@/lib/content";
import { classByKey } from "@/lib/classes";
import { classifyImage, decodeImage, loadModel, type Prediction } from "@/lib/inference";
import { ResultPanel } from "./ResultPanel";

const MAX_UPLOAD_BYTES = 20 * 1024 * 1024;

type ModelState = "idle" | "loading" | "ready" | "error";
type RunState =
  | { kind: "empty" }
  | { kind: "running" }
  | { kind: "done"; prediction: Prediction }
  | { kind: "error"; message: string };

interface Selected {
  src: string;
  title: string;
  trueLabel?: string;
  objectUrl?: boolean;
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : "Something went wrong while analysing the image.";
}

export function Analyzer() {
  const [modelState, setModelState] = useState<ModelState>("idle");
  const [selected, setSelected] = useState<Selected | null>(null);
  const [run, setRun] = useState<RunState>({ kind: "empty" });
  const [group, setGroup] = useState<SampleGroup | "all">("all");
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const runId = useRef(0);

  const ensureModel = useCallback(async () => {
    setModelState((s) => (s === "ready" ? s : "loading"));
    try {
      await loadModel();
      setModelState("ready");
    } catch (err) {
      setModelState("error");
      throw err;
    }
  }, []);

  // Warm the model once the browser is idle so the first click is fast.
  useEffect(() => {
    const warm = () => ensureModel().catch(() => undefined);
    if ("requestIdleCallback" in window) {
      const id = window.requestIdleCallback(warm, { timeout: 2500 });
      return () => window.cancelIdleCallback(id);
    }
    const t = setTimeout(warm, 1200);
    return () => clearTimeout(t);
  }, [ensureModel]);

  // Release object URLs from previous uploads.
  useEffect(() => {
    return () => {
      if (selected?.objectUrl) URL.revokeObjectURL(selected.src);
    };
  }, [selected]);

  const analyse = useCallback(
    async (next: Selected) => {
      const id = ++runId.current;
      setSelected(next);
      setRun({ kind: "running" });
      try {
        await ensureModel();
        const image = await decodeImage(next.src);
        const prediction = await classifyImage(image);
        if (id === runId.current) setRun({ kind: "done", prediction });
      } catch (err) {
        if (id === runId.current) setRun({ kind: "error", message: errorMessage(err) });
      }
    },
    [ensureModel],
  );

  const handleFile = useCallback(
    (file: File | undefined) => {
      if (!file) return;
      if (!file.type.startsWith("image/")) {
        setRun({ kind: "error", message: "Please choose an image file (JPEG, PNG or WebP)." });
        return;
      }
      if (file.size > MAX_UPLOAD_BYTES) {
        setRun({ kind: "error", message: "That image is larger than 20 MB — please use a smaller one." });
        return;
      }
      analyse({ src: URL.createObjectURL(file), title: file.name, objectUrl: true });
    },
    [analyse],
  );

  const pickSample = (sample: Sample) =>
    analyse({ src: sample.src, title: `${classByKey(sample.label).short} · ${sample.source}`, trueLabel: sample.label });

  const visibleSamples = group === "all" ? SAMPLES : SAMPLES.filter((s) => sampleGroup(s) === group);

  return (
    <section id="analyse" className="bg-sand py-20 md:py-28">
      <div className="mx-auto max-w-6xl px-4 md:px-8">
        <div className="max-w-2xl">
          <p className="eyebrow">Analyse</p>
          <h2 className="mt-3 font-serif text-4xl leading-tight text-ink md:text-6xl">
            Upload an image, or start with a sample.
          </h2>
          <p className="mt-4 text-muted">
            Best results come from a sharp, well-lit close-up of a single lesion — ideally taken through a
            dermatoscope. Your image is analysed entirely in this browser and never uploaded anywhere.
          </p>
        </div>

        <div className="mt-10 grid gap-6 lg:grid-cols-2">
          {/* Input / preview */}
          <div
            className={`relative flex min-h-[22rem] flex-col items-center justify-center overflow-hidden rounded-[2rem] border-2 border-dashed bg-cream p-6 text-center transition-colors ${
              dragging ? "border-clay-deep bg-linen" : "border-line"
            }`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              handleFile(e.dataTransfer.files?.[0]);
            }}
          >
            {selected ? (
              <figure className="flex w-full flex-col items-center gap-4">
                {/* eslint-disable-next-line @next/next/no-img-element -- object URLs and local samples */}
                <img
                  src={selected.src}
                  alt={selected.title}
                  className="aspect-square w-full max-w-sm rounded-[1.5rem] object-cover shadow-sm"
                />
                <figcaption className="text-sm text-muted">{selected.title}</figcaption>
              </figure>
            ) : (
              <div className="flex flex-col items-center gap-3">
                <span className="flex h-14 w-14 items-center justify-center rounded-full bg-linen text-2xl text-clay-deep" aria-hidden>
                  ↑
                </span>
                <p className="font-serif text-2xl text-ink">Drop an image here</p>
                <p className="text-sm text-muted">JPEG, PNG or WebP · up to 20 MB</p>
              </div>
            )}
            <div className="mt-6 flex flex-wrap justify-center gap-3">
              <button
                type="button"
                onClick={() => inputRef.current?.click()}
                className="rounded-full bg-ink px-6 py-3 text-sm font-medium text-cream transition hover:bg-ink-soft focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ink"
              >
                {selected ? "Choose another image" : "Choose an image"}
              </button>
            </div>
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              className="sr-only"
              data-testid="file-input"
              aria-label="Upload a skin image"
              onChange={(e) => {
                handleFile(e.target.files?.[0]);
                e.target.value = "";
              }}
            />
          </div>

          {/* Result */}
          <div className="min-h-[22rem] rounded-[2rem] bg-cream p-6 md:p-8" aria-live="polite">
            {run.kind === "empty" && (
              <div className="flex h-full flex-col justify-center gap-3">
                <p className="eyebrow">Result</p>
                <p className="font-serif text-3xl text-ink">Your analysis will appear here.</p>
                <p className="text-sm text-muted">
                  Model status:{" "}
                  <span className="font-medium text-text">
                    {modelState === "ready" && "ready"}
                    {modelState === "loading" && "loading (one-time download)…"}
                    {modelState === "idle" && "waiting"}
                    {modelState === "error" && "failed to load — check your connection and try again"}
                  </span>
                </p>
              </div>
            )}
            {run.kind === "running" && (
              <div className="flex h-full flex-col items-center justify-center gap-4 text-muted" role="status">
                <span className="h-10 w-10 animate-spin rounded-full border-2 border-line border-t-clay-deep" aria-hidden />
                {modelState === "loading" ? "Loading the model…" : "Analysing…"}
              </div>
            )}
            {run.kind === "error" && (
              <div className="flex h-full flex-col justify-center gap-2" role="alert">
                <p className="eyebrow text-sev-malignant">Couldn&apos;t analyse</p>
                <p className="text-text">{run.message}</p>
              </div>
            )}
            {run.kind === "done" && <ResultPanel prediction={run.prediction} trueLabel={selected?.trueLabel} />}
          </div>
        </div>

        {/* Samples */}
        <div id="samples" className="mt-16">
          <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
            <div>
              <p className="eyebrow">Sample images</p>
              <h3 className="mt-2 font-serif text-3xl text-ink">Test the model on images it has never seen.</h3>
              <p className="mt-2 max-w-xl text-sm text-muted">
                Every sample comes from a held-out test set — lesions from the official ISIC 2018 test set, skin
                conditions from Google&apos;s SCIN dataset, and random everyday images and textures.
              </p>
            </div>
            <div className="flex flex-wrap gap-2" role="tablist" aria-label="Filter samples">
              {[{ key: "all" as const, label: "All" }, ...SAMPLE_GROUPS].map((g) => (
                <button
                  key={g.key}
                  type="button"
                  role="tab"
                  aria-selected={group === g.key}
                  onClick={() => setGroup(g.key)}
                  className={`rounded-full border px-4 py-1.5 text-sm transition ${
                    group === g.key ? "border-ink bg-ink text-cream" : "border-line bg-cream text-text hover:border-clay"
                  }`}
                >
                  {g.label}
                </button>
              ))}
            </div>
          </div>

          <ul className="mt-8 grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6">
            {visibleSamples.map((sample) => {
              const info = classByKey(sample.label);
              const active = selected?.src === sample.src;
              return (
                <li key={sample.id}>
                  <button
                    type="button"
                    onClick={() => pickSample(sample)}
                    data-testid="sample"
                    className={`group block w-full overflow-hidden rounded-2xl border bg-cream text-left transition hover:-translate-y-0.5 hover:shadow-md focus-visible:outline-2 focus-visible:outline-ink ${
                      active ? "border-ink ring-2 ring-ink" : "border-line"
                    }`}
                  >
                    {/* eslint-disable-next-line @next/next/no-img-element -- small pre-sized local thumbnails */}
                    <img src={sample.src} alt={info.short} loading="lazy" className="aspect-square w-full object-cover" />
                    <span className="block px-3 py-2">
                      <span className="block truncate text-sm font-medium">{info.short}</span>
                      <span className="block truncate text-xs text-muted">{sample.source}</span>
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        </div>
      </div>
    </section>
  );
}
