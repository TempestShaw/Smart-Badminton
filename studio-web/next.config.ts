import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",
  assetPrefix: "/static",
  images: { unoptimized: true },
  poweredByHeader: false,
  generateBuildId: async () => "smart-badminton-studio",
};

export default nextConfig;
