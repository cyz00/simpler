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

// Two-rank HBM AllGather (ccu_id 0) and AllToAll (ccu_id 1).
// The runtime already consumed ready/trigger. This entry loads P and calls the
// pto-isa data path. P layout, domain rank order:
//   for each rank: send address, send token, recv address, recv token
//   length, scratch
//   AllToAll only: blockOffset[0], blockOffset[1]
//   addrOffset, loopIterNum, parallelParam, residual
// AllGather is 14 words (112 bytes). AllToAll is 16 words (128 bytes).

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "hcomm/ccu/ccu_primitives.hpp"
#include "hcomm/ccu/ccu_res.h"
#include "platform_comm/ccu/ccu_native_abi.h"
#include "pto/comm/async/ccu/ccu_collective_ops.hpp"

extern const uint8_t ccu_test_sdk_identity[32];
extern const uint8_t ccu_test_isa_identity[32];

namespace {
namespace ccu = AscendC::ccu;
namespace pccu = pto::comm::ccu;

constexpr uint32_t kRanks = 2;
constexpr uint32_t kGatherWords = 14;
constexpr uint32_t kAllToAllWords = 16;
constexpr uint32_t kParallelChunks = 2;
constexpr uint32_t kChunkBytes = 32u * 1024u;
constexpr uint64_t kPayloadBytes = 64ull * 1024ull;
constexpr uint32_t kGatherBytes = kGatherWords * sizeof(uint64_t);
constexpr uint32_t kAllToAllBytes = kAllToAllWords * sizeof(uint64_t);

void check(CcuResult rc, const char *what) {
    if (rc != CCU_SUCCESS) throw std::runtime_error(what);
}

// This entry runs several times per mission — the resource sizing query, the
// registration, and whatever passes the SDK makes internally — so the same
// line repeating for one rank and ccu_id is expected.
void report_path(
    const SimplerCcuRegistrationBinding &binding, const char *op, const char *path, uint32_t word_count
) {
    const uint32_t banks = std::strcmp(path, "ms") == 0 ? kChunkBytes / 4096u : 0u;
    std::fprintf(
        stderr, "[ccu_collective] rank %u ccu_id %u %s path=%s params=%u words chunk=%u B ms_banks=%u\n",
        binding.rank_id, binding.ccu_id, op, path, word_count, kChunkBytes, banks
    );
}

int32_t collective(const SimplerCcuRegistrationBinding *binding, bool all_to_all) noexcept {
    try {
        if (binding == nullptr || binding->rank_count != kRanks || binding->rank_id >= kRanks) {
            throw std::runtime_error("CCU collective v1 requires exactly 2 ranks");
        }
        // CcuAllGatherOp reads PTO_CCU_AG_BACKEND here and treats every value
        // other than "ms" as HBM, so a variable that never reached this forked
        // process would quietly pick HBM. Make the path an explicit choice.
        const char *backend = ::getenv("PTO_CCU_AG_BACKEND");
        if (backend == nullptr || (std::strcmp(backend, "hbm") != 0 && std::strcmp(backend, "ms") != 0)) {
            throw std::runtime_error("PTO_CCU_AG_BACKEND must name hbm or ms");
        }
        const uint32_t word_count = all_to_all ? kAllToAllWords : kGatherWords;
        std::array<ccu::Variable, SIMPLER_CCU_TASK_ARGUMENT_COUNT> launch;
        for (uint32_t i = 0; i < launch.size(); ++i) check(ccu::LoadArg(launch[i], i), "LoadArg failed");

        // One variable-address load. The byte offsets are baked into that
        // instruction, so this entry never multiplies on the CCU.
        ccu::Array<ccu::Variable> words(word_count);
        check(ccu::Load(launch[0], words, word_count), "Load parameter failed");

        pccu::CcuStageRuntimeVars stage;
        stage.sliceInput.resize(kRanks);
        stage.sliceOutput.resize(kRanks);
        stage.token.resize(kRanks);
        for (uint32_t rank = 0; rank < kRanks; ++rank) {
            stage.sliceInput[rank] = words[4u * rank];
            stage.token[rank] = words[4u * rank + 1u];
            stage.sliceOutput[rank] = words[4u * rank + 2u];
        }
        stage.lengthVar = words[8];
        stage.sliceScratch = words[9];
        const uint32_t go = all_to_all ? 12u : 10u;
        if (all_to_all) {
            stage.blockOffset.resize(kRanks);
            stage.blockOffset[0] = words[10];
            stage.blockOffset[1] = words[11];
        }
        stage.transferAddrOffset = words[go];
        stage.transferLoopCount = words[go + 1u];
        stage.transferParallelParam = words[go + 2u];
        stage.transferResidual = words[go + 3u];

        std::vector<ChannelHandle> channels;
        channels.reserve(kRanks - 1u);
        for (uint32_t rank = 0; rank < kRanks; ++rank) {
            if (rank == binding->rank_id) continue;
            channels.push_back(binding->channels[rank]);
        }

        pccu::CcuLoopGroupResourcePool pool;
        if (all_to_all) {
            pccu::CcuAllToAllOp op(pccu::CcuAllToAllKernelArg(binding->rank_id, kRanks, kPayloadBytes));
            if (op.Init(kParallelChunks, kChunkBytes, pool) != HcclResult::HCCL_SUCCESS) {
                throw std::runtime_error("AllToAll Init failed");
            }
            report_path(*binding, "AllToAll", "hbm", word_count);
            op.Emit(stage, channels);
        } else {
            pccu::CcuAllGatherOp op(pccu::CcuAllGatherKernelArg(binding->rank_id, kRanks, kPayloadBytes));
            if (op.Init(kParallelChunks, kChunkBytes, pool) != HcclResult::HCCL_SUCCESS) {
                throw std::runtime_error("AllGather Init failed");
            }
            // Ask the same predicate the op constructor used, so the line
            // reports the emitted path and not just the environment string.
            report_path(*binding, "AllGather", pccu::AllGatherUseMsBackend() ? "ms" : "hbm", word_count);
            op.Emit(stage, channels);
        }
        ccu::Event done(binding->done_event_handle);
        check(ccu::EventRecord(done), "EventRecord done failed");
        return CCU_SUCCESS;
    } catch (...) {
        return CCU_E_INTERNAL;
    }
}

constexpr char kGatherSymbol[] = "ccu_allgather";
constexpr char kAllToAllSymbol[] = "ccu_alltoall";

struct Manifest {
    SimplerCcuModuleManifest header;
    SimplerCcuKernelRecord kernels[2];
    char symbols[sizeof(kGatherSymbol) - 1 + sizeof(kAllToAllSymbol) - 1];
};

Manifest make_manifest() {
    Manifest manifest{};
    manifest.header.struct_size = sizeof(manifest.header);
    manifest.header.manifest_size = sizeof(manifest);
    manifest.header.kernel_record_size = sizeof(manifest.kernels[0]);
    manifest.header.kernel_count = 2;
    manifest.header.kernel_records_offset = offsetof(Manifest, kernels);
    manifest.header.string_table_offset = offsetof(Manifest, symbols);
    manifest.header.string_table_size = sizeof(manifest.symbols);
    std::memcpy(manifest.header.sdk_build_id, ccu_test_sdk_identity, 32);
    std::memcpy(manifest.header.pto_isa_build_id, ccu_test_isa_identity, 32);
    std::memcpy(manifest.symbols, kGatherSymbol, sizeof(kGatherSymbol) - 1);
    std::memcpy(manifest.symbols + sizeof(kGatherSymbol) - 1, kAllToAllSymbol, sizeof(kAllToAllSymbol) - 1);
    const uint32_t sizes[] = {kGatherBytes, kAllToAllBytes};
    const uint32_t symbol_sizes[] = {sizeof(kGatherSymbol) - 1, sizeof(kAllToAllSymbol) - 1};
    uint32_t symbol_offset = 0;
    for (uint32_t i = 0; i < 2; ++i) {
        manifest.kernels[i] = {
            sizeof(SimplerCcuKernelRecord),
            i,
            sizes[i],
            64,
            SIMPLER_CCU_KERNEL_CAP_MASK,
            manifest.header.string_table_offset + symbol_offset,
            symbol_sizes[i],
            sizeof(SimplerCcuRegistrationBinding),
            SIMPLER_CCU_REGISTRATION_ARGUMENT_COUNT,
            SIMPLER_CCU_TASK_ARGUMENT_COUNT
        };
        symbol_offset += symbol_sizes[i];
    }
    return manifest;
}
}  // namespace

extern "C" int32_t ccu_allgather(const SimplerCcuRegistrationBinding *binding) noexcept {
    return collective(binding, false);
}
extern "C" int32_t ccu_alltoall(const SimplerCcuRegistrationBinding *binding) noexcept {
    return collective(binding, true);
}
extern "C" const SimplerCcuModuleManifest *simpler_ccu_module_manifest() noexcept {
    static const auto manifest = make_manifest();
    return &manifest.header;
}
