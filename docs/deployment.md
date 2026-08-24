# Online deployment

## Decision

The browser uploads source videos directly to object storage. A control API owns projects and jobs, while GPU workers run the existing analysis and rendering pipeline. Local Studio and hosted Studio share one project manifest and the same pipeline entrypoint; storage is the only adapter.

```text
Next.js Studio
  -> control API + database
  -> presigned object-storage upload
  -> durable GPU job queue
  -> Hybrid analysis / FFmpeg worker
  -> object storage results
```

The web frontend may be hosted separately. Video upload, CUDA inference, FFmpeg rendering and model files do not run inside the frontend host's request functions.

## Required changes

1. Replace the process-wide active video with user-scoped `project_id` requests.
2. Replace folder selection with resumable upload and a project list.
3. Store source video, proxy, analysis cache and exports by object key instead of browser-provided paths.
4. Move analysis state from in-process threads to durable queued jobs with retry, cancellation and progress.
5. Package the Hybrid worker and its pinned CUDA/FFmpeg dependencies as a container.
6. Add authentication, ownership checks, quotas, retention and deletion controls.

## Delivery order

- **Gate 0 — licensing:** confirm redistribution and hosted-inference rights for every checkpoint and runtime.
- **Milestone 1 — cloud project contract:** introduce project, artifact and job records without changing pipeline algorithms.
- **Milestone 2 — upload and preview:** direct multipart upload, proxy job and browser playback.
- **Milestone 3 — GPU pipeline:** queued Hybrid analysis, timeline save and render output.
- **Milestone 4 — production controls:** authentication, billing limits, observability, backups and deletion.
- **Milestone 5 — CI/CD:** test gates, frontend preview deployment, worker image publication and production promotion.

## Invariants

- Human timelines and score labels are durable source data; predictions never overwrite them.
- Large media bypasses the control API and moves through signed object-storage URLs.
- Jobs may resume after a process or worker restart.
- Every artifact records the pipeline version, model checksums and source-video checksum.
- The UI never receives server filesystem paths or model-provider credentials.
