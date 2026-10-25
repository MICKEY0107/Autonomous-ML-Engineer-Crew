import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, "..", "");

  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        "/api": env.VITE_API_PROXY_TARGET,
      },
    },
    build: {
      target: "es2020",
      sourcemap: false,
      // Recharts is isolated behind React.lazy; the chart chunk is ~151 KB gzip.
      chunkSizeWarningLimit: 700,
      rollupOptions: {
        output: {
          manualChunks: {
            charts: ["recharts"],
            motion: ["framer-motion"],
          },
        },
      },
    },
  };
});
