import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Two ways this app runs, and both are covered here:
//
//   1. Production/demo — `npm run build` emits dist/, and Flask serves it from
//      the same origin as the API. Relative /api/... URLs just work.
//   2. Development — `npm run dev` on 5173 proxies /api, /webhook and
//      /agent-manifest through to Flask on 5050, so the frontend code can use
//      the same relative URLs in both modes and never needs a base-URL env var.
const BACKEND = 'http://127.0.0.1:5050'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': BACKEND,
      '/webhook': BACKEND,
      '/agent-manifest': BACKEND,
      '/catalog.json': BACKEND,
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
})
