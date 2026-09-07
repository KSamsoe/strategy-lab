import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The console is a local-first single-user app: `lab ui` serves the built SPA
// from FastAPI on 127.0.0.1. In dev we run Vite separately and proxy /api to
// the same backend, so the frontend never needs a second source of truth for
// where the lab lives.
const LAB_API = process.env.LAB_API_URL ?? "http://127.0.0.1:8787";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    outDir: "dist",
    sourcemap: true,
    chunkSizeWarningLimit: 1200,
  },
  server: {
    port: 5173,
    strictPort: false,
    proxy: {
      "/api": { target: LAB_API, changeOrigin: true, ws: true },
    },
  },
});
// Vitest options deliberately live in package.json's test script rather than
// here: vitest ships its own copy of vite, and a `test` block in this file
// makes the two vite type trees collide during `tsc -b`.
