import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// Drives the real React app against a RUNNING backend. See docs/LOCAL_SETUP.md.
export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "./src/test/setup.js",
    css: false,
    include: ["src/e2e/**/*.e2e.test.jsx"],
    testTimeout: 180000,
    hookTimeout: 60000,
    fileParallelism: false,
  },
});
