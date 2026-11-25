// MoE routing trace for expert-usage measurement (SQUID_EXPERT_TRACE=<path>).
// One 16-byte record per routed token per layer visit, appended in stream order to a
// zero-copy pinned host buffer. The append index lives in device memory; launches on one
// stream are serialized, so the records come out in execution order. d_count[1] holds the
// request index, read at run time because decode launches are replayed from CUDA graphs.

#include <cuda_runtime.h>
#include <cstdint>

struct ExpertTraceRecord {
    int32_t visit;      // 0..47 layer visit
    int32_t tag;        // request << 20 | phase << 16 | token index in the batch (phase 0 = decode/b8, 1 = batched prefill)
    uint8_t ids[8];     // top-8 expert ids
};
static_assert(sizeof(ExpertTraceRecord) == 16, "trace record must stay 16 bytes");

__global__ void expert_trace_kernel(const int32_t* __restrict__ route_ids, int B, int visit, int phase,
                                    int32_t* __restrict__ d_count, ExpertTraceRecord* __restrict__ out, int cap) {
    __shared__ int base;
    if (threadIdx.x == 0) {
        base = *d_count;
        *d_count = base + B;
    }
    __syncthreads();
    for (int t = threadIdx.x; t < B; t += blockDim.x) {
        const int r = base + t;
        if (r >= cap) continue;
        ExpertTraceRecord rec;
        rec.visit = visit;
        rec.tag = (d_count[1] << 20) | (phase << 16) | t;
#pragma unroll
        for (int k = 0; k < 8; ++k) rec.ids[k] = static_cast<uint8_t>(route_ids[t * 8 + k]);
        out[r] = rec;
    }
}

extern "C" void squid_launch_expert_trace(const int32_t* route_ids, int32_t B, int32_t visit, int32_t phase,
                                          int32_t* d_count, void* host_dev_ptr, int32_t cap, void* stream) {
    expert_trace_kernel<<<1, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        route_ids, B, visit, phase, d_count, static_cast<ExpertTraceRecord*>(host_dev_ptr), cap);
}
