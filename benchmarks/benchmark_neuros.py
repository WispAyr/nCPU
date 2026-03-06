#!/usr/bin/env python3
"""neurOS Benchmarks.

Measures performance of all neurOS components:
    - Boot time
    - MMU translation throughput
    - TLB hit/miss latency
    - Cache access latency
    - Scheduler decision latency
    - Filesystem operations
    - Assembler throughput
    - Compiler throughput
    - End-to-end compilation pipeline
"""

import sys
import time
import torch

sys.path.insert(0, ".")
from ncpu.os import NeurOS


def benchmark_boot(n: int = 5):
    """Benchmark boot time."""
    times = []
    for _ in range(n):
        os = NeurOS()
        t = time.perf_counter()
        os.boot(quiet=True)
        times.append(time.perf_counter() - t)
    avg = sum(times) / len(times)
    mn = min(times)
    print(f"  Boot time:       {avg*1000:8.1f}ms avg  {mn*1000:.1f}ms min  (n={n})")
    return times


def benchmark_mmu(os, n: int = 1000):
    """Benchmark MMU translation."""
    # Map some pages first
    for vpn in range(100):
        os.mmu.map_page(vpn, vpn + 1000, read=True, write=True)

    t = time.perf_counter()
    for i in range(n):
        os.mmu.translate(i % 100)
    elapsed = time.perf_counter() - t
    print(f"  MMU translate:   {elapsed/n*1e6:8.1f}us/op  ({n} ops in {elapsed*1000:.1f}ms)")


def benchmark_tlb(os, n: int = 1000):
    """Benchmark TLB lookups."""
    # Warm up
    perms = torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.float32, device=os.device)
    for vpn in range(50):
        os.tlb.insert(vpn, 0, vpn + 1000, perms)

    t = time.perf_counter()
    hits = 0
    for i in range(n):
        result = os.tlb.lookup(i % 64)
        if result is not None:
            hits += 1
    elapsed = time.perf_counter() - t
    print(f"  TLB lookup:      {elapsed/n*1e6:8.1f}us/op  (hit rate: {hits/n:.0%})")


def benchmark_cache(os, n: int = 1000):
    """Benchmark cache access."""
    t = time.perf_counter()
    hits = 0
    for i in range(n):
        addr = (i % 200) * 64
        if os.cache.access(addr):
            hits += 1
    elapsed = time.perf_counter() - t
    print(f"  Cache access:    {elapsed/n*1e6:8.1f}us/op  (hit rate: {hits/n:.0%})")


def benchmark_scheduler(os, n: int = 100):
    """Benchmark scheduler decisions."""
    from ncpu.os.process import ProcessState
    # Create some processes
    for i in range(10):
        p = os.process_table.create_process(f"bench_{i}", priority=i % 5)
        p.state = ProcessState.READY
        os.ipc.register_process(p.pid)

    t = time.perf_counter()
    for _ in range(n):
        os.scheduler.schedule()
    elapsed = time.perf_counter() - t
    print(f"  Scheduler:       {elapsed/n*1e6:8.1f}us/op  ({n} scheduling decisions)")


def benchmark_filesystem(os, n: int = 100):
    """Benchmark filesystem operations."""
    # Write
    data = torch.tensor([65] * 100, dtype=torch.uint8, device=os.device)
    t = time.perf_counter()
    for i in range(n):
        os.fs.write_file(f"/tmp/bench_{i}", data)
    write_time = time.perf_counter() - t

    # Read
    t = time.perf_counter()
    for i in range(n):
        os.fs.read_file(f"/tmp/bench_{i}")
    read_time = time.perf_counter() - t

    # Stat
    t = time.perf_counter()
    for i in range(n):
        os.fs.stat(f"/tmp/bench_{i}")
    stat_time = time.perf_counter() - t

    print(f"  FS write:        {write_time/n*1e6:8.1f}us/op")
    print(f"  FS read:         {read_time/n*1e6:8.1f}us/op")
    print(f"  FS stat:         {stat_time/n*1e6:8.1f}us/op")


def benchmark_assembler(os, n: int = 50):
    """Benchmark assembler throughput."""
    source = """\
    MOV R0, 0
    MOV R1, 1
    MOV R2, 10
loop:
    ADD R0, R0, R1
    ADD R1, R1, R1
    CMP R1, R2
    JNZ loop
    HALT
"""
    t = time.perf_counter()
    for _ in range(n):
        os.assembler.assemble(source)
    elapsed = time.perf_counter() - t
    result = os.assembler.assemble(source)
    print(f"  Assembler:       {elapsed/n*1e6:8.1f}us/prog  "
          f"({result.num_instructions} instr, {n} programs)")


def benchmark_compiler(os, n: int = 50):
    """Benchmark compiler throughput."""
    source = """\
var sum = 0;
var i = 1;
var limit = 11;
while (i != limit) {
    sum = sum + i;
    i = i + 1;
}
halt;
"""
    t = time.perf_counter()
    for _ in range(n):
        os.compiler.compile(source)
    elapsed = time.perf_counter() - t
    result = os.compiler.compile(source)
    asm_count = result.assembly_result.num_instructions if result.assembly_result else 0
    print(f"  Compiler:        {elapsed/n*1e6:8.1f}us/prog  "
          f"({len(result.ir)} IR -> {asm_count} asm, {n} programs)")


def benchmark_end_to_end(os):
    """Benchmark full compile → assemble → binary pipeline."""
    source = """\
var a = 7;
var b = 6;
var result = a * b;
halt;
"""
    t = time.perf_counter()
    result = os.compiler.compile(source)
    compile_time = time.perf_counter() - t
    binary = result.binary
    print(f"  End-to-end:      {compile_time*1e6:8.1f}us  "
          f"(nsl -> {len(result.ir)} IR -> {len(binary)} binary words)")


def main():
    print("=" * 60)
    print("  neurOS Performance Benchmarks")
    print("=" * 60)
    print()

    # Boot
    print("[Boot]")
    benchmark_boot()
    print()

    # Create OS instance for component benchmarks
    os = NeurOS()
    os.boot(quiet=True)

    print("[Memory Subsystem]")
    benchmark_mmu(os)
    benchmark_tlb(os)
    benchmark_cache(os)
    print()

    print("[Process Management]")
    benchmark_scheduler(os)
    print()

    print("[Filesystem]")
    benchmark_filesystem(os)
    print()

    print("[Toolchain]")
    benchmark_assembler(os)
    benchmark_compiler(os)
    benchmark_end_to_end(os)
    print()

    print("=" * 60)


if __name__ == "__main__":
    main()
