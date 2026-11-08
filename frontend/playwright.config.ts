import { defineConfig, devices } from "@playwright/test";

const PORT = 3300;
/** Set E2E_BASE_URL (e.g. the Vercel production URL) to test a deployed site instead of a local build. */
const REMOTE_URL = process.env.E2E_BASE_URL;

export default defineConfig({
  testDir: "./e2e",
  timeout: 90_000,
  fullyParallel: false,
  reporter: "list",
  use: {
    baseURL: REMOTE_URL ?? `http://localhost:${PORT}`,
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: REMOTE_URL
    ? undefined
    : {
        command: `npm run build && npx next start -p ${PORT}`,
        url: `http://localhost:${PORT}`,
        timeout: 180_000,
        reuseExistingServer: !process.env.CI,
      },
});
