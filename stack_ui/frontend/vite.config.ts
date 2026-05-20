import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

/** Dev proxy target for `/api` → Stack UI backend. Set in `.env.local` (gitignored) or env. */
function resolveApiTarget(mode: string): string {
  const fromFile = loadEnv(mode, process.cwd(), "").STACK_UI_API_TARGET?.trim();
  const fromEnv = process.env.STACK_UI_API_TARGET?.trim();
  return fromEnv || fromFile || "http://127.0.0.1:8765";
}

export default defineConfig(({ mode }) => {
  const apiTarget = resolveApiTarget(mode);
  return {
    plugins: [react()],
    server: {
      host: "0.0.0.0",
      port: 5173,
      proxy: {
        "/api": {
          target: apiTarget,
          changeOrigin: true,
        },
      },
    },
  };
});
