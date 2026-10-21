import adapter from '@sveltejs/adapter-static';

export default {
  kit: {
    adapter: adapter({ pages: 'build', assets: 'build', strict: true }),
    files: { assets: 'public' },
    paths: { relative: true }
  }
};
