/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${process.env.NEXT_PUBLIC_API_URL || 'http://54.87.245.227:8000'}/api/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
