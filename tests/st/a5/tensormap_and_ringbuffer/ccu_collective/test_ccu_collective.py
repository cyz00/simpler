# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Two-rank HBM AllGather and AllToAll through the simpler CCU runtime."""

import hashlib
import os
import struct
import subprocess
from pathlib import Path

import pytest
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import ChipCallable, CommBufferSpec, CoreCallable, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler

_ROOT = Path(__file__).resolve().parent
_RUNTIME = "tensormap_and_ringbuffer"
_RANKS = 2
_BLOCK = 64 * 1024
_RECV = _RANKS * _BLOCK
_AG_SEND = 0
_AG_RECV = _BLOCK
_A2A_SEND = 256 * 1024
_A2A_RECV = _A2A_SEND + _RECV
_WINDOW = 1024 * 1024
# 2 lanes x 32 KiB, one even iteration of a 64 KiB block. n == 0 and p == 0.
_GO = (_BLOCK, 1, 0, 0)


def _collective_isa_root():
    """Use the pto-isa tree that has host-written AllToAll offsets.

    The pinned checkout does not. SIMPLER_CCU_PTO_ISA overrides the sibling
    checkout next to this simpler tree.
    """
    override = os.environ.get("SIMPLER_CCU_PTO_ISA")
    root = Path(override) if override else Path(__file__).resolve().parents[6] / "pto-isa"
    header = root / "include/pto/comm/async/ccu/ccu_collective_ops.hpp"
    text = header.read_text(encoding="utf-8")
    if "host-written block offset" not in text:
        raise RuntimeError(
            f"{root} does not contain the host-written AllToAll block offsets. "
            "Point SIMPLER_CCU_PTO_ISA at that pto-isa tree."
        )
    return root


def _build_callable(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    sdk = Path(os.environ["ASCEND_HOME_PATH"])
    arch_name = "aarch64-linux" if (sdk / "aarch64-linux").is_dir() else "x86_64-linux"
    arch = sdk / arch_name
    common = Path(__file__).resolve().parents[5] / "src/common"
    isa_root = _collective_isa_root()
    sdk_identity = hashlib.sha256((arch / "include/hcomm/ccu/ccu_res.h").read_bytes()).digest()
    isa_identity = hashlib.sha256((isa_root / "include/pto/pto-inst.hpp").read_bytes()).digest()
    identities = tmp_path / "identities.cpp"
    identities.write_text(
        "#include <cstdint>\n"
        f"extern const uint8_t ccu_test_sdk_identity[32] = {{{','.join(map(str, sdk_identity))}}};\n"
        f"extern const uint8_t ccu_test_isa_identity[32] = {{{','.join(map(str, isa_identity))}}};\n"
    )
    module = tmp_path / "native_module.so"
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-O2",
            str(_ROOT / "native_module.cpp"),
            str(identities),
            f"-I{common}",
            f"-I{isa_root / 'include'}",
            f"-I{arch / 'include'}",
            f"-I{arch / 'include/hccl'}",
            f"-I{arch / 'pkg_inc/hcomm/ccu'}",
            f"-I{arch / 'include/hcomm/ccu'}",
            f"-L{sdk / 'lib64'}",
            "-lasccomm_ccu",
            "-lhcomm",
            "-o",
            str(module),
        ],
        check=True,
    )
    compiler = KernelCompiler(platform="a5")
    orchestration = compiler.compile_orchestration(
        _RUNTIME, str(_ROOT / "kernels/orchestration/ccu_orch.cpp")
    )
    includes = compiler.get_orchestration_include_dirs(_RUNTIME)
    submit = compiler.compile_incore(
        str(_ROOT / "kernels/aiv/submit.cpp"),
        core_type="aiv",
        pto_isa_root=str(isa_root),
        extra_include_dirs=includes,
    )
    copy = compiler.compile_incore(
        str(_ROOT / "kernels/aiv/copy_recv.cpp"),
        core_type="aiv",
        pto_isa_root=str(isa_root),
        extra_include_dirs=includes,
    )
    # Scalars count as signature entries: the submit task takes the status
    # tensor plus nine scalars, the copy task the status and evidence tensors
    # plus three.
    submit_signature = [D.OUT] + [D.IN] * 9
    copy_signature = [D.IN, D.OUT] + [D.IN] * 3
    children = [
        (0, CoreCallable.build(signature=submit_signature, binary=extract_text_section(submit))),
        (1, CoreCallable.build(signature=copy_signature, binary=extract_text_section(copy))),
    ]
    return ChipCallable.build(
        signature=[D.OUT, D.OUT, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN],
        func_name="ccu_collective_orchestration",
        config_name="ccu_collective_orchestration_config",
        binary=orchestration,
        children=children,
        ccu_modules=[module.read_bytes()],
    )


def _fill(rank, blocks):
    payload = bytearray()
    for block in range(blocks):
        payload += bytes([(rank * 16 + block + 1) & 0xFF]) * _BLOCK
    return bytes(payload)


def _expected_allgather():
    return b"".join(_fill(rank, 1) for rank in range(_RANKS))


def _expected_alltoall(dest):
    pieces = []
    for src in range(_RANKS):
        send = _fill(src, _RANKS)
        pieces.append(send[dest * _BLOCK : (dest + 1) * _BLOCK])
    return b"".join(pieces)


@pytest.mark.platforms(["a5"])
@pytest.mark.runtime(_RUNTIME)
@pytest.mark.device_count(2)
@pytest.mark.parametrize("ag_backend", ["hbm", "ms"])
def test_ccu_allgather_and_alltoall(st_platform, st_device_ids, tmp_path, monkeypatch, ag_backend):
    # CcuAllGatherOp picks its data path when it is constructed, which happens
    # in the chip process while register_ccu_callable emits the mission. init()
    # forks that process, so the choice has to be in the environment already.
    # AllToAll has one path and ignores the variable.
    monkeypatch.setenv("PTO_CCU_AG_BACKEND", ag_backend)
    target = _build_callable(tmp_path)
    worker = Worker(
        level=3,
        platform=st_platform,
        runtime=_RUNTIME,
        device_ids=[int(device) for device in st_device_ids[:2]],
        num_sub_workers=0,
    )
    try:
        chip = worker.register(target)
        worker.init()
        worker.register_ccu_callable(chip)
        status = [worker.create_buffer(64) for _ in range(_RANKS)]
        evidence = [worker.create_buffer(_RECV) for _ in range(_RANKS)]
        domain = {}

        def run_one(mode, send_off, recv_off, payloads):
            def run(orch, _args, config):
                handle = orch.get_or_allocate_domain(
                    name="ccu_collective",
                    members=[0, 1],
                    window_size=_WINDOW,
                    buffers=[CommBufferSpec("data", "uint64", _WINDOW // 8, _WINDOW)],
                    workspace_engines=("ccu",),
                    retain_after_run=True,
                )
                domain["handle"] = handle
                for rank in range(_RANKS):
                    orch.copy_to_domain(handle, rank, payloads[rank], buffer="data", offset=send_off)
                    orch.copy_to_domain(handle, rank, bytes(_RECV), buffer="data", offset=recv_off)
                for rank in range(_RANKS):
                    args = TaskArgs()
                    args.add_tensor(status[rank].tensor((8,), DataType.UINT64), TensorArgType.OUTPUT_EXISTING)
                    args.add_tensor(
                        evidence[rank].tensor((_RECV // 8,), DataType.UINT64), TensorArgType.OUTPUT_EXISTING
                    )
                    args.add_scalar(handle.device_context(rank, workspace_engines=("ccu",)))
                    args.add_scalar(mode)
                    args.add_scalar(_BLOCK)
                    args.add_scalar(send_off)
                    args.add_scalar(recv_off)
                    for word in _GO:
                        args.add_scalar(word)
                    orch.submit_next_level(chip, args, config, worker=rank)

            worker.run(run)
            for rank in range(_RANKS):
                submitted, nbytes, got_rank, got_mode = struct.unpack("4Q", bytes(status[rank].shm.buf[:32]))
                assert (submitted, nbytes, got_rank, got_mode) == (1, 112 if mode == 0 else 128, rank, mode)

        run_one(0, _AG_SEND, _AG_RECV, [_fill(rank, 1) for rank in range(_RANKS)])
        expected = _expected_allgather()
        for rank in range(_RANKS):
            assert bytes(evidence[rank].shm.buf[:_RECV]) == expected, f"allgather rank {rank}"

        run_one(1, _A2A_SEND, _A2A_RECV, [_fill(rank, _RANKS) for rank in range(_RANKS)])
        for rank in range(_RANKS):
            assert bytes(evidence[rank].shm.buf[:_RECV]) == _expected_alltoall(rank), f"alltoall rank {rank}"
        handle = domain.get("handle")
        if handle is not None:
            handle.release()
    finally:
        worker.close()
