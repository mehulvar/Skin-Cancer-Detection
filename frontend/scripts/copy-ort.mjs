// Copies the ONNX Runtime Web WebAssembly binaries into public/ort so they are
// served from our own origin (required for cross-origin isolation + threads).
import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const src = join(root, "node_modules", "onnxruntime-web", "dist");
const dest = join(root, "public", "ort");
const FILES = ["ort-wasm-simd-threaded.wasm", "ort-wasm-simd-threaded.mjs"];

mkdirSync(dest, { recursive: true });
for (const file of FILES) {
  const from = join(src, file);
  if (!existsSync(from)) {
    console.error(`copy-ort: missing ${from} — is onnxruntime-web installed?`);
    process.exit(1);
  }
  copyFileSync(from, join(dest, file));
}
console.log(`copy-ort: copied ${FILES.length} files to public/ort`);
