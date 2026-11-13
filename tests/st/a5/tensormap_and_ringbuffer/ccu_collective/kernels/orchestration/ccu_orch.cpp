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

#include "orchestration_api.h"

extern "C" __attribute__((visibility("default"))) OrchestrationConfig ccu_collective_orchestration_config(
    const ChipTaskArgs &
) {
    return OrchestrationConfig{.expected_arg_count = 11};
}

extern "C" __attribute__((visibility("default"))) void ccu_collective_orchestration(const ChipTaskArgs &args) {
    auto status = args.tensor(0).ref();
    auto evidence = args.tensor(1).ref();
    const auto context = args.scalar<uint64_t>(0);
    const auto mode = args.scalar<uint64_t>(1);
    const auto block = args.scalar<uint64_t>(2);
    const auto send_off = args.scalar<uint64_t>(3);
    const auto recv_off = args.scalar<uint64_t>(4);
    const auto go0 = args.scalar<uint64_t>(5);
    const auto go1 = args.scalar<uint64_t>(6);
    const auto go2 = args.scalar<uint64_t>(7);
    const auto go3 = args.scalar<uint64_t>(8);

    CoreTaskArgs submit;
    submit.add_output(status);
    submit.add_scalar(context);
    submit.add_scalar(mode);
    submit.add_scalar(block);
    submit.add_scalar(send_off);
    submit.add_scalar(recv_off);
    submit.add_scalar(go0);
    submit.add_scalar(go1);
    submit.add_scalar(go2);
    submit.add_scalar(go3);
    rt_submit_aiv_task(0, submit);

    CoreTaskArgs copy;
    copy.add_input(status);
    copy.add_output(evidence);
    copy.add_scalar(context);
    copy.add_scalar(recv_off);
    copy.add_scalar(block * 2);
    rt_submit_aiv_task(1, copy);
}
