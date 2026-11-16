import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: "#0b1016",
        panel: "#121923",
        card: "#18212c",
        line: "#293544",
      },
      fontFamily: {
        sans: ["Inter", "Segoe UI", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "Cascadia Code", "monospace"],
      },
      boxShadow: {
        panel: "0 18px 48px rgba(0, 0, 0, 0.28)",
        glow: "0 0 0 1px rgba(6,182,212,.3), 0 0 24px rgba(6,182,212,.12)",
      },
    },
  },
  plugins: [],
} satisfies Config;
