/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#pragma once

#include <pto/pto-inst.hpp>
#include "async_kernel_api.h"
#include "platform_comm/comm_context.h"
#include "tensor.h"

// Wire layout the CCU entry reads back, in domain rank order: per rank the send
// address and token then the recv address and token, the length, the scratch
// slot, AllToAll's two source block offsets, and the four GoSize words.
// AllGather is 14 words, AllToAll 16.
inline __aicore__ bool submit_allgather(
    AsyncCtx &ctx, const __gm__ CommContext *comm, uint64_t block, uint64_t send_off, uint64_t recv_off, uint64_t go0,
    uint64_t go1, uint64_t go2, uint64_t go3
) {
    using Arg = simpler::ccu::CcuCallArg;
    const uint64_t recv_bytes = block * 2;
    const Arg call_args[]{
        {Arg::Kind::Window, 0, send_off, block},
        {Arg::Kind::Window, 0, recv_off, recv_bytes},
        {Arg::Kind::Window, 1, send_off, block},
        {Arg::Kind::Window, 1, recv_off, recv_bytes},
        {Arg::Kind::Scalar, 0, block, 0},
        {Arg::Kind::Scalar, 0, comm->rankId * block, 0},
        {Arg::Kind::Scalar, 0, go0, 0},
        {Arg::Kind::Scalar, 0, go1, 0},
        {Arg::Kind::Scalar, 0, go2, 0},
        {Arg::Kind::Scalar, 0, go3, 0},
    };
    return submit_ccu_mission(ctx, comm, 0u, call_args);
}

inline __aicore__ bool submit_alltoall(
    AsyncCtx &ctx, const __gm__ CommContext *comm, uint64_t block, uint64_t send_off, uint64_t recv_off, uint64_t go0,
    uint64_t go1, uint64_t go2, uint64_t go3
) {
    using Arg = simpler::ccu::CcuCallArg;
    const uint64_t bytes = block * 2;
    const Arg call_args[]{
        {Arg::Kind::Window, 0, send_off, bytes},
        {Arg::Kind::Window, 0, recv_off, bytes},
        {Arg::Kind::Window, 1, send_off, bytes},
        {Arg::Kind::Window, 1, recv_off, bytes},
        {Arg::Kind::Scalar, 0, block, 0},
        {Arg::Kind::Scalar, 0, comm->rankId * block, 0},
        {Arg::Kind::Scalar, 0, 0, 0},
        {Arg::Kind::Scalar, 0, block, 0},
        {Arg::Kind::Scalar, 0, go0, 0},
        {Arg::Kind::Scalar, 0, go1, 0},
        {Arg::Kind::Scalar, 0, go2, 0},
        {Arg::Kind::Scalar, 0, go3, 0},
    };
    return submit_ccu_mission(ctx, comm, 1u, call_args);
}

// args[0] status, args[1] comm, args[2] mode, args[3] block bytes,
// args[4] send offset, args[5] recv offset, args[6..9] GoSize.
inline __aicore__ void submit_collective(__gm__ int64_t *args) {
    constexpr uint32_t kRanks = 2;
    auto *status = reinterpret_cast<__gm__ Tensor *>(args[0]);
    const auto *comm = reinterpret_cast<const __gm__ CommContext *>(args[1]);
    const uint64_t mode = static_cast<uint64_t>(args[2]);
    const uint64_t block = static_cast<uint64_t>(args[3]);
    const uint64_t send_off = static_cast<uint64_t>(args[4]);
    const uint64_t recv_off = static_cast<uint64_t>(args[5]);
    const uint64_t go0 = static_cast<uint64_t>(args[6]);
    const uint64_t go1 = static_cast<uint64_t>(args[7]);
    const uint64_t go2 = static_cast<uint64_t>(args[8]);
    const uint64_t go3 = static_cast<uint64_t>(args[9]);
    auto *out = reinterpret_cast<__gm__ uint64_t *>(status->buffer.addr) + status->start_offset;
    const bool ranks_ok = comm != nullptr && comm->rankNum == kRanks && comm->rankId < kRanks;
    bool submitted = false;
    if (ranks_ok && block != 0 && mode == 0) {
        AsyncCtx ctx = get_async_ctx(args);
        submitted = submit_allgather(ctx, comm, block, send_off, recv_off, go0, go1, go2, go3);
    } else if (ranks_ok && block != 0 && mode == 1) {
        AsyncCtx ctx = get_async_ctx(args);
        submitted = submit_alltoall(ctx, comm, block, send_off, recv_off, go0, go1, go2, go3);
    }
    out[0] = submitted ? 1 : 0;
    out[1] = mode == 0 ? 112 : 128;
    out[2] = ranks_ok ? comm->rankId : 0xffffffffu;
    out[3] = mode;
    dcci(out, cache_line_t::ENTIRE_DATA_CACHE);
    dsb(DSB_DDR);
}
