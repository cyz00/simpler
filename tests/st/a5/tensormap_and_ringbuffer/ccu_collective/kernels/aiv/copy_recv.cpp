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

#include <pto/pto-inst.hpp>
#include "platform_comm/comm_context.h"
#include "tensor.h"

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    auto *evidence = reinterpret_cast<__gm__ Tensor *>(args[1]);
    const auto *comm = reinterpret_cast<const __gm__ CommContext *>(args[2]);
    const uint64_t recv_off = static_cast<uint64_t>(args[3]);
    const uint64_t nbytes = static_cast<uint64_t>(args[4]);
    const uint64_t count = nbytes / sizeof(uint64_t);
    auto *source = reinterpret_cast<__gm__ uint64_t *>(comm->windowsIn[comm->rankId] + recv_off);
    auto *out = reinterpret_cast<__gm__ uint64_t *>(evidence->buffer.addr) + evidence->start_offset;
    dcci(source, cache_line_t::ENTIRE_DATA_CACHE);
    for (uint64_t i = 0; i < count; ++i) out[i] = source[i];
    dcci(out, cache_line_t::ENTIRE_DATA_CACHE);
    dsb(DSB_DDR);
}
