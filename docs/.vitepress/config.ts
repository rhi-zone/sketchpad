import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

export default withMermaid(
  defineConfig({
    title: 'sketchpad',
    description: 'Stable Diffusion inference in pure Rust with Burn',

    base: '/sketchpad/',

    srcExclude: ['**/CLAUDE.md'],

    themeConfig: {
      nav: [
        { text: 'Guide', link: '/' },
        { text: 'Samplers', link: '/samplers' },
        { text: 'Pipelines', link: '/pipelines' },
        { text: 'rhi', link: 'https://docs.rhi.zone/' },
      ],

      sidebar: [
        {
          text: 'Guide',
          items: [
            { text: 'Introduction', link: '/' },
            { text: 'Getting Started', link: '/getting-started' },
            { text: 'Architecture', link: '/architecture' },
          ]
        },
        {
          text: 'Reference',
          items: [
            { text: 'Pipelines', link: '/pipelines' },
            { text: 'Samplers', link: '/samplers' },
            { text: 'Future Architectures', link: '/future-architectures' },
          ]
        },
      ],

      socialLinks: [
        { icon: 'github', link: 'https://github.com/rhi-zone/sketchpad' }
      ],

      search: {
        provider: 'local'
      },

      editLink: {
        pattern: 'https://github.com/rhi-zone/sketchpad/edit/master/docs/:path',
        text: 'Edit this page on GitHub'
      },
    },

    vite: {
      optimizeDeps: {
        include: ['mermaid'],
      },
    },
  }),
)
