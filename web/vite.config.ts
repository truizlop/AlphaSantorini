import { defineConfig } from "vite";

const buildId = process.env.GITHUB_SHA ?? String(Date.now());

export default defineConfig({
  base: "./",
  define: {
    __BUILD_ID__: JSON.stringify(buildId),
  },
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "credentialless",
    },
  },
});
