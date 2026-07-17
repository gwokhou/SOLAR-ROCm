# ROCm evaluation image

Build the pinned gfx1200 evaluator with:

```bash
docker build -f docker/Dockerfile -t solar-rocm:7.2 .
```

`solar-evaluate --untrusted` mounts benchmark and solution packages read-only,
passes `/dev/kfd` and `/dev/dri`, verifies `STABLE_PEAK` on the host, and writes
only the requested `evaluation.yaml` to the output mount.
