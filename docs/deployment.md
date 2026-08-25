# Online deployment

## Production architecture

The public Next.js export runs **Browser Quick** without an application backend:

```text
Static Next.js page
  -> browser File object
  -> local frame sampling and segmentation
  -> local timeline storage
  -> local FFmpeg WebAssembly export
```

The source video is not uploaded. Hosting serves HTML, JavaScript and the pinned FFmpeg WebAssembly runtime, so there is no per-video GPU bill.

**Native Accurate** remains a local companion. The web page links to the current GitHub Release and opens `127.0.0.1:8765` after the user starts the native Studio. CUDA, Hybrid models, full trajectory analysis and native FFmpeg remain outside the hosted page.

## Builds

- `npm run build` creates the deployable web export in `studio-web/out/`.
- `npm run build:embedded` publishes the `/static` build into `smart_badminton/studio_static/` for FastAPI.
- Both builds share React components, segment types and timeline semantics.

## Deployment

Configure the Vercel project root as `studio-web`, then deploy the static export. `studio-web/vercel.json` applies immutable caching to FFmpeg assets. No environment variables, storage account or database are required.

## Boundaries

- Browser Quick uses lightweight adaptive visual activity, not the full Hybrid detector.
- Browser export currently accepts source files up to 1.5 GB because FFmpeg WebAssembly uses browser memory.
- Native Accurate is the supported path for long recordings, full trajectories, scoring and CUDA.
- Manual timelines and score labels remain local source data and are never overwritten by predictions.
