/**
 * Image → model input tensor.
 *
 * Mirrors training/skinlesion/data.py::center_square_resize + Normalize:
 * centre-crop to a square on the short side, bilinear-resize to `size`,
 * scale to [0,1], subtract ImageNet mean and divide by std, CHW layout.
 */

export type Rgb = readonly [number, number, number];

export interface PreprocessConfig {
  size: number;
  mean: Rgb;
  std: Rgb;
}

export interface SquareCrop {
  sx: number;
  sy: number;
  side: number;
}

export function centerSquareCrop(width: number, height: number): SquareCrop {
  if (width <= 0 || height <= 0) {
    throw new Error("Image has no pixels");
  }
  const side = Math.min(width, height);
  return {
    sx: Math.floor((width - side) / 2),
    sy: Math.floor((height - side) / 2),
    side,
  };
}

/** Convert RGBA pixel bytes (size×size) into a normalised CHW Float32Array. */
export function rgbaToTensor(
  rgba: Uint8ClampedArray,
  { size, mean, std }: PreprocessConfig,
): Float32Array {
  const plane = size * size;
  if (rgba.length !== plane * 4) {
    throw new Error(`Expected ${plane * 4} RGBA bytes, got ${rgba.length}`);
  }
  const out = new Float32Array(plane * 3);
  for (let i = 0; i < plane; i++) {
    const p = i * 4;
    out[i] = (rgba[p] / 255 - mean[0]) / std[0];
    out[plane + i] = (rgba[p + 1] / 255 - mean[1]) / std[1];
    out[2 * plane + i] = (rgba[p + 2] / 255 - mean[2]) / std[2];
  }
  return out;
}

/** Horizontally mirror a CHW tensor (used for flip test-time augmentation). */
export function flipHorizontal(tensor: Float32Array, size: number): Float32Array {
  const out = new Float32Array(tensor.length);
  const channels = tensor.length / (size * size);
  for (let c = 0; c < channels; c++) {
    const base = c * size * size;
    for (let y = 0; y < size; y++) {
      const row = base + y * size;
      for (let x = 0; x < size; x++) {
        out[row + x] = tensor[row + size - 1 - x];
      }
    }
  }
  return out;
}

/** Draw an image source into a size×size canvas using the centre-square crop. */
export function imageToTensor(
  source: CanvasImageSource & { width: number; height: number },
  config: PreprocessConfig,
): Float32Array {
  const { sx, sy, side } = centerSquareCrop(source.width, source.height);
  const canvas = document.createElement("canvas");
  canvas.width = config.size;
  canvas.height = config.size;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) {
    throw new Error("Canvas 2D context is unavailable in this browser");
  }
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(source, sx, sy, side, side, 0, 0, config.size, config.size);
  const { data } = ctx.getImageData(0, 0, config.size, config.size);
  return rgbaToTensor(data, config);
}
