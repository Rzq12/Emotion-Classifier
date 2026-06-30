/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        paper: "#f6f5f1",
        surface: "#ffffff",
        ink: "#17181c",
        muted: "#6c6a63",
        faint: "#9b988f",
        line: "#e4e1d8",
        // Emotion semantic palette (always paired with a label in UI).
        anger: "#bf3b30",
        sadness: "#3f6694",
        happiness: "#2f8a6a",
        focus: "#3f6694",
      },
      fontFamily: {
        display: ['Fraunces', 'Georgia', 'serif'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'ui-monospace', 'monospace'],
      },
      keyframes: {
        "fade-up": {
          "0%": { opacity: "0", transform: "translateY(6px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        "fade-up": "fade-up 0.35s ease-out both",
      },
    },
  },
  plugins: [],
};
