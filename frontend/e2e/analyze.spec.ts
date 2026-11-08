import { expect, test, type Page } from "@playwright/test";
import path from "node:path";
import samples from "../src/data/samples.json" with { type: "json" };

const FIXTURES = path.join(__dirname, "fixtures");

async function waitForResult(page: Page) {
  const result = page.getByTestId("result");
  await expect(result).toBeVisible({ timeout: 60_000 });
  return result;
}

test("page is cross-origin isolated for multi-threaded inference", async ({ page }) => {
  await page.goto("/");
  expect(await page.evaluate(() => crossOriginIsolated)).toBe(true);
  await expect(page.getByRole("heading", { level: 1 })).toContainText("Clarity");
});

test("every sample image is classified as its ground-truth label", async ({ page }) => {
  await page.goto("/");
  const buttons = page.getByTestId("sample");
  await expect(buttons).toHaveCount(samples.length);

  for (let i = 0; i < samples.length; i++) {
    await buttons.nth(i).click();
    const result = await waitForResult(page);
    await expect(result).toHaveAttribute("data-prediction", samples[i].label, { timeout: 30_000 });
    await expect(result).toContainText("✓ correct");
  }
});

test("a random non-skin upload is rejected as 'Not skin'", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("file-input").setInputFiles(path.join(FIXTURES, "random-noise.png"));
  const result = await waitForResult(page);
  await expect(result).toHaveAttribute("data-prediction", "not_skin");
  await expect(result).toContainText("This doesn't look like skin");
  await expect(result).not.toContainText("Ground truth");
});

test("a non-image file shows a friendly error", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("file-input").setInputFiles(path.join(FIXTURES, "not-an-image.txt"));
  await expect(page.getByRole("alert").filter({ hasText: "Please choose an image file" })).toBeVisible();
});

test("sample filters narrow the gallery", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("tab", { name: "Not skin" }).click();
  const expected = samples.filter((s) => s.label === "not_skin").length;
  await expect(page.getByTestId("sample")).toHaveCount(expected);
});
