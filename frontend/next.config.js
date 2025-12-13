/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8000/api/:path*',
      },
      {
        source: '/images/:path*',
        destination: 'http://localhost:8000/images/:path*',
      },
      {
        source: '/catalog/:path*',
        destination: 'http://localhost:8000/catalog/:path*',
      },
    ]
  },
}

module.exports = nextConfig
