import type { NextConfig } from "next";

const ONE_DAY = 60 * 60 * 24;

const nextConfig: NextConfig = {
  cacheComponents: true,
  partialPrefetching: true,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
  async headers() {
    return [
      {
        // Cross-origin isolation enables multi-threaded WebAssembly inference.
        source: "/:path*",
        headers: [
          { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
          { key: "Cross-Origin-Embedder-Policy", value: "require-corp" },
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
        ],
      },
      {
        source: "/(model|ort|samples)/:file*",
        headers: [{ key: "Cache-Control", value: `public, max-age=${ONE_DAY}, stale-while-revalidate=${ONE_DAY * 30}` }],
      },
    ];
  },
};

export default nextConfig;
