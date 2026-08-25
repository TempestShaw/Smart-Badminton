import type { NextConfig } from "next";

const embeddedBuild = process.env.STUDIO_EMBEDDED_BUILD === "1";

const nextConfig: NextConfig = {
  output: "export",
  assetPrefix: embeddedBuild ? "/static" : undefined,
  images: { unoptimized: true },
  poweredByHeader: false,
  generateBuildId: async () => embeddedBuild ? "smart-badminton-studio" : "smart-badminton-web",
};

export default nextConfig;
