# ROCm evaluation image

Build the pinned ROCm 7.2 evaluator with:

```bash
docker build -f docker/Dockerfile -t solar-rocm:7.2 .
```

`solar-evaluate --untrusted` mounts benchmark and solution packages read-only,
passes `/dev/kfd` and `/dev/dri`, verifies `STABLE_PEAK` on the host, and writes
only the requested `evaluation.yaml` to the output mount. The evaluator checks
the detected `gfx*` target against the selected architecture profile and the
solution manifest; it does not override the HSA-reported architecture.
