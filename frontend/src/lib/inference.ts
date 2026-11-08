/**
 * In-browser inference with onnxruntime-web (WebAssembly).
 * The image never leaves the device: the model is downloaded once and run locally.
 */

import type { InferenceSession } from "onnxruntime-web";
import { CLASSES } from "./classes";
import { averageProbabilities, interpret, softmax, type Verdict } from "./interpret";
import { flipHorizontal, imageToTensor, type PreprocessConfig } from "./preprocess";

export const MODEL_URL = "/model/skin_lesion.onnx";
export const WASM_URL = "/ort/ort-wasm-simd-threaded.wasm";
const MAX_THREADS = 4;

export const PREPROCESS: PreprocessConfig = {
  size: 224,
  mean: [0.485, 0.456, 0.406],
  std: [0.229, 0.224, 0.225],
};

export interface Prediction {
  verdict: Verdict;
  probabilities: number[];
  elapsedMs: number;
}

let sessionPromise: Promise<InferenceSession> | null = null;

async function createSession(): Promise<InferenceSession> {
  const ort = await import("onnxruntime-web/wasm");
  ort.env.wasm.wasmPaths = { wasm: WASM_URL };
  const isolated = typeof crossOriginIsolated !== "undefined" && crossOriginIsolated;
  ort.env.wasm.numThreads = isolated
    ? Math.min(MAX_THREADS, navigator.hardwareConcurrency || 1)
    : 1;
  return ort.InferenceSession.create(MODEL_URL, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
}

/** Load (or reuse) the model session. Safe to call repeatedly; retries after a failure. */
export function loadModel(): Promise<InferenceSession> {
  if (!sessionPromise) {
    sessionPromise = createSession().catch((err: unknown) => {
      sessionPromise = null;
      throw err;
    });
  }
  return sessionPromise;
}

async function runLogits(session: InferenceSession, data: Float32Array): Promise<number[]> {
  const ort = await import("onnxruntime-web/wasm");
  const { size } = PREPROCESS;
  const input = new ort.Tensor("float32", data, [1, 3, size, size]);
  const output = await session.run({ [session.inputNames[0]]: input });
  const logits = output[session.outputNames[0]];
  return Array.from(logits.data as Float32Array);
}

/** Classify an already-decoded image. Averages the image and its mirror (flip TTA). */
export async function classifyImage(
  image: CanvasImageSource & { width: number; height: number },
): Promise<Prediction> {
  const session = await loadModel();
  const start = performance.now();
  const tensor = imageToTensor(image, PREPROCESS);
  const plain = softmax(await runLogits(session, tensor));
  const flipped = softmax(await runLogits(session, flipHorizontal(tensor, PREPROCESS.size)));
  const probabilities = averageProbabilities([plain, flipped]);
  return {
    verdict: interpret(probabilities, CLASSES),
    probabilities,
    elapsedMs: performance.now() - start,
  };
}

/** Decode an image URL (object URL or same-origin path) into an ImageBitmap-like element. */
export function decodeImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.decoding = "async";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("This file could not be read as an image."));
    img.src = src;
  });
}
