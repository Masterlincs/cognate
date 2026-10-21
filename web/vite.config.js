import adapter from '@sveltejs/adapter-static';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [
    sveltekit({
      adapter: adapter({ pages: 'build', assets: 'build', strict: true }),
      files: { assets: 'public' },
      paths: { relative: true },
      inlineStyleThreshold: 16384
    })
  ],
  test: { include: ['src/**/*.test.js'] }
});
