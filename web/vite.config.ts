import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const target = 'http://localhost:8000'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Backend runs on :8000; proxied so same-origin fetches just work in dev.
    proxy: {
      '/livekit': target,
      '/health': target,
      '/query': target, // also covers /query/stream
    },
  },
})
