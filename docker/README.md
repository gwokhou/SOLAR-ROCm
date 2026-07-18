# ROCm evaluation image

Build the pinned ROCm 7.2 evaluator with:

```bash
docker build -f docker/Dockerfile -t solar-rocm:7.2 .
```

This is the single delivery image. A pinned Ubuntu builder stage statically
builds the exact Orojenesis/Timeloop mapper revision and emits a hash-bound
`/opt/orojenesis/orojenesis-provenance.json`; only that binary and manifest are
copied into the ROCm/PyTorch final stage. The builder, source tree, and any
NVIDIA runtime are absent from the final image.

Run the real mapper contract stage independently with:

```bash
docker build --target orojenesis-test -f docker/Dockerfile .
```

It executes single-einsum, canonical multi-einsum, and extended-region mapper
proofs and fails if the pinned toolchain is missing or cannot be replayed.

`solar-evaluate --untrusted` mounts benchmark and solution packages read-only,
passes `/dev/kfd` and `/dev/dri`, verifies `STABLE_PEAK` on the host, and writes
only the requested `evaluation.yaml` to the output mount. The evaluator checks
the detected `gfx*` target against the selected architecture profile and the
solution manifest; it does not override the HSA-reported architecture.
