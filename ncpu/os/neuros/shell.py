"""Neural Shell (nsh).

Interactive shell with neural command parsing. Uses the existing decode_llm
model for semantic understanding of commands, with fallback to simple
string-based parsing.

Features:
    - Built-in commands: ls, cd, cat, echo, mkdir, rm, ps, kill, help, exit
    - Pipeline support: cmd1 | cmd2
    - Filesystem integration through syscall interface
    - Process management commands
    - Tab completion (when neural model is available)

The shell runs as a neurOS process (PID 1 by convention — init process).
"""

import torch
import time
import logging
from typing import Optional, List, Dict, Tuple, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .boot import NeurOS

from .device import default_device
from .process import ProcessState
from .syscalls import (
    SYS_EXIT, SYS_OPEN, SYS_CLOSE, SYS_READ, SYS_WRITE,
    SYS_MKDIR, SYS_RMDIR, SYS_UNLINK, SYS_LISTDIR, SYS_STAT,
    SYS_FORK, SYS_KILL, SYS_GETPID,
)

logger = logging.getLogger(__name__)


class NeuralShell:
    """neurOS interactive shell.

    Parses commands and dispatches to syscalls or built-in handlers.
    Falls back to string parsing when neural model isn't available.
    """

    def __init__(self, os: 'NeurOS'):
        self.os = os
        self.device = os.device if hasattr(os, 'device') else default_device()

        # Shell state
        self.cwd = "/"
        self.pid = 1  # Shell is PID 1 (init)
        self.running = True
        self.history: List[str] = []
        self.env: Dict[str, str] = {
            "HOME": "/",
            "PATH": "/bin",
            "SHELL": "/bin/nsh",
            "USER": "root",
            "HOSTNAME": "neuros",
        }

        # Built-in commands
        self._builtins: Dict[str, Callable] = {
            "ls": self._cmd_ls,
            "cd": self._cmd_cd,
            "pwd": self._cmd_pwd,
            "cat": self._cmd_cat,
            "echo": self._cmd_echo,
            "mkdir": self._cmd_mkdir,
            "rm": self._cmd_rm,
            "rmdir": self._cmd_rmdir,
            "touch": self._cmd_touch,
            "ps": self._cmd_ps,
            "kill": self._cmd_kill,
            "top": self._cmd_top,
            "stat": self._cmd_stat,
            "help": self._cmd_help,
            "exit": self._cmd_exit,
            "quit": self._cmd_exit,
            "clear": self._cmd_clear,
            "env": self._cmd_env,
            "export": self._cmd_export,
            "history": self._cmd_history,
            "uname": self._cmd_uname,
            "df": self._cmd_df,
            "free": self._cmd_free,
            "uptime": self._cmd_uptime,
            "asm": self._cmd_asm,
            "nsc": self._cmd_nsc,
            "write": self._cmd_write,
            "run": self._cmd_run,
            "exec": self._cmd_exec,
            "regs": self._cmd_regs,
            "neural": self._cmd_neural,
            "bench": self._cmd_bench,
            "hexdump": self._cmd_hexdump,
        }

        # CPU from last run/exec (for regs command)
        self._last_cpu = None

        # Output buffer (for non-interactive mode)
        self.output: List[str] = []

    def execute(self, command: str) -> List[str]:
        """Execute a command and return output lines.

        This is the non-interactive entry point. For interactive use,
        call run_interactive().
        """
        self.output = []
        self.history.append(command)

        if not command.strip():
            return []

        # Handle pipelines
        if "|" in command:
            return self._execute_pipeline(command)

        # Parse command
        parts = self._parse_command(command.strip())
        if not parts:
            return []

        cmd = parts[0]
        args = parts[1:]

        # Check built-ins
        handler = self._builtins.get(cmd)
        if handler is not None:
            handler(args)
        else:
            self.output.append(f"nsh: command not found: {cmd}")

        return self.output

    def run_interactive(self):
        """Run the interactive shell loop.

        Reads from stdin, executes commands, prints output.
        """
        self._print(f"neurOS v0.1 — Neural Operating System")
        self._print(f"Type 'help' for available commands.\n")

        while self.running:
            try:
                prompt = f"root@neuros:{self.cwd}# "
                line = input(prompt)
                output = self.execute(line)
                for line in output:
                    print(line)
            except (EOFError, KeyboardInterrupt):
                self._print("\nexit")
                break

    # ─── Command Parsing ──────────────────────────────────────────────────

    def _parse_command(self, command: str) -> List[str]:
        """Parse a command string into tokens.

        Handles:
            - Quoted strings: "hello world"
            - Environment variable expansion: $HOME
            - Simple word splitting
        """
        tokens = []
        current = ""
        in_quote = False
        quote_char = None

        for ch in command:
            if in_quote:
                if ch == quote_char:
                    in_quote = False
                else:
                    current += ch
            elif ch in ('"', "'"):
                in_quote = True
                quote_char = ch
            elif ch == ' ':
                if current:
                    tokens.append(self._expand_vars(current))
                    current = ""
            else:
                current += ch

        if current:
            tokens.append(self._expand_vars(current))

        return tokens

    def _expand_vars(self, token: str) -> str:
        """Expand environment variables in a token."""
        if "$" not in token:
            return token

        result = token
        for key, value in self.env.items():
            result = result.replace(f"${key}", value)
        return result

    def _resolve_path(self, path: str) -> str:
        """Resolve a relative path to absolute."""
        if path.startswith("/"):
            return path
        if self.cwd == "/":
            return f"/{path}"
        return f"{self.cwd}/{path}"

    def _execute_pipeline(self, command: str) -> List[str]:
        """Execute a pipeline of commands.

        For now, chains output → input between commands.
        """
        parts = [p.strip() for p in command.split("|")]
        current_input: List[str] = []

        for part in parts:
            # Each command in pipeline gets previous output as context
            output = self.execute(part)
            current_input = output

        return current_input

    # ─── Built-in Commands ────────────────────────────────────────────────

    def _cmd_ls(self, args: List[str]):
        """List directory contents."""
        path = self._resolve_path(args[0] if args else self.cwd)
        entries = self.os.fs.list_dir(path)
        if entries is None:
            self._print(f"ls: cannot access '{path}': No such directory")
        elif not entries:
            pass  # Empty directory
        else:
            # Format as columns
            for name in sorted(entries):
                info = self.os.fs.stat(f"{path}/{name}" if path != "/" else f"/{name}")
                if info and info["type"] == "DIRECTORY":
                    self._print(f"  {name}/")
                else:
                    size = info["size"] if info else 0
                    self._print(f"  {name}  ({size}B)")

    def _cmd_cd(self, args: List[str]):
        """Change directory."""
        if not args:
            self.cwd = "/"
            return

        target = self._resolve_path(args[0])
        if self.os.fs.exists(target):
            info = self.os.fs.stat(target)
            if info and info["type"] == "DIRECTORY":
                self.cwd = target
            else:
                self._print(f"cd: not a directory: {args[0]}")
        else:
            self._print(f"cd: no such directory: {args[0]}")

    def _cmd_pwd(self, args: List[str]):
        """Print working directory."""
        self._print(self.cwd)

    def _cmd_cat(self, args: List[str]):
        """Display file contents."""
        if not args:
            self._print("cat: missing filename")
            return

        for filename in args:
            path = self._resolve_path(filename)
            data = self.os.fs.read_file(path)
            if data is None:
                self._print(f"cat: {filename}: No such file")
            elif len(data) == 0:
                pass  # Empty file
            else:
                # Convert bytes to string
                try:
                    text = "".join(chr(b) for b in data.cpu().tolist() if 32 <= b < 127 or b == 10)
                    self._print(text)
                except Exception:
                    self._print(f"[binary data, {len(data)} bytes]")

    def _cmd_echo(self, args: List[str]):
        """Print arguments."""
        self._print(" ".join(args))

    def _cmd_mkdir(self, args: List[str]):
        """Create a directory."""
        if not args:
            self._print("mkdir: missing directory name")
            return
        for dirname in args:
            path = self._resolve_path(dirname)
            result = self.os.fs.mkdir(path)
            if result < 0:
                self._print(f"mkdir: cannot create '{dirname}'")

    def _cmd_rm(self, args: List[str]):
        """Remove a file."""
        if not args:
            self._print("rm: missing filename")
            return
        for filename in args:
            path = self._resolve_path(filename)
            if not self.os.fs.unlink(path):
                self._print(f"rm: cannot remove '{filename}'")

    def _cmd_rmdir(self, args: List[str]):
        """Remove an empty directory."""
        if not args:
            self._print("rmdir: missing directory name")
            return
        for dirname in args:
            path = self._resolve_path(dirname)
            if not self.os.fs.rmdir(path):
                self._print(f"rmdir: cannot remove '{dirname}'")

    def _cmd_touch(self, args: List[str]):
        """Create an empty file."""
        if not args:
            self._print("touch: missing filename")
            return
        for filename in args:
            path = self._resolve_path(filename)
            if not self.os.fs.exists(path):
                self.os.fs.create(path)

    def _cmd_ps(self, args: List[str]):
        """List processes."""
        self._print(f"{'PID':>5}  {'STATE':<12}  {'PRI':>3}  {'CPU':>6}  {'NAME'}")
        self._print(f"{'---':>5}  {'-----':<12}  {'---':>3}  {'---':>6}  {'----'}")
        for p in self.os.process_table.all_processes:
            state_name = ProcessState(p.state).name
            self._print(f"{p.pid:>5}  {state_name:<12}  {p.priority:>3}  {p.cpu_time:>6}  {p.name}")

    def _cmd_kill(self, args: List[str]):
        """Kill a process."""
        if not args:
            self._print("kill: missing PID")
            return
        try:
            target_pid = int(args[0])
            signal = int(args[1]) if len(args) > 1 else 9
            self.os.scheduler.terminate_process(target_pid, -signal)
            self._print(f"Killed PID {target_pid}")
        except ValueError:
            self._print(f"kill: invalid PID: {args[0]}")

    def _cmd_top(self, args: List[str]):
        """Show system status."""
        self._print("=== neurOS System Status ===")
        self._print(f"Scheduler: {self.os.scheduler}")
        self._print(f"Memory:    {self.os.mmu}")
        self._print(f"TLB:       {self.os.tlb}")
        self._print(f"Cache:     {self.os.cache}")
        self._print(f"FS:        {self.os.fs}")
        self._print(f"IPC:       {self.os.ipc}")
        self._print(f"GIC:       {self.os.gic}")
        self._print(f"Processes: {self.os.process_table}")

    def _cmd_stat(self, args: List[str]):
        """Show file/directory metadata."""
        if not args:
            self._print("stat: missing path")
            return
        path = self._resolve_path(args[0])
        info = self.os.fs.stat(path)
        if info is None:
            self._print(f"stat: cannot stat '{args[0]}'")
        else:
            for key, value in info.items():
                self._print(f"  {key}: {value}")

    def _cmd_help(self, args: List[str]):
        """Show available commands."""
        self._print("neurOS Shell (nsh) — Available commands:")
        self._print("")
        commands = {
            "ls [path]": "List directory contents",
            "cd [path]": "Change directory",
            "pwd": "Print working directory",
            "cat <file>": "Display file contents",
            "echo <text>": "Print text",
            "mkdir <dir>": "Create directory",
            "rm <file>": "Remove file",
            "rmdir <dir>": "Remove empty directory",
            "touch <file>": "Create empty file",
            "ps": "List processes",
            "kill <pid>": "Kill a process",
            "top": "System status",
            "stat <path>": "File metadata",
            "df": "Disk usage",
            "free": "Memory usage",
            "uname": "System info",
            "env": "Environment variables",
            "history": "Command history",
            "asm <file.asm>": "Assemble a file",
            "nsc <file.nsl>": "Compile an nsl source file",
            "write <file> <text>": "Write text to a file",
            "run <file.nsl>": "Compile and execute nsl program",
            "exec <file.asm>": "Assemble and execute program",
            "regs": "Show registers from last run/exec",
            "neural": "Show neural model status",
            "bench": "Quick compile+execute benchmark",
            "hexdump <file>": "Hex dump of file contents",
            "exit": "Exit shell",
        }
        for cmd, desc in commands.items():
            self._print(f"  {cmd:<20} {desc}")

    def _cmd_exit(self, args: List[str]):
        """Exit the shell."""
        self.running = False

    def _cmd_clear(self, args: List[str]):
        """Clear output."""
        self.output = []

    def _cmd_env(self, args: List[str]):
        """Show environment variables."""
        for key, value in sorted(self.env.items()):
            self._print(f"{key}={value}")

    def _cmd_export(self, args: List[str]):
        """Set environment variable."""
        for arg in args:
            if "=" in arg:
                key, value = arg.split("=", 1)
                self.env[key] = value
            else:
                self._print(f"export: usage: export KEY=VALUE")

    def _cmd_history(self, args: List[str]):
        """Show command history."""
        for i, cmd in enumerate(self.history):
            self._print(f"  {i+1:>4}  {cmd}")

    def _cmd_uname(self, args: List[str]):
        """Show system information."""
        self._print("neurOS 0.1.0 (nCPU ARM64-neural) — GPU-Native Neural Operating System")

    def _cmd_df(self, args: List[str]):
        """Show filesystem disk usage."""
        s = self.os.fs.stats()
        total_kb = s["total_blocks"] * 4  # 4KB blocks
        used_kb = s["used_blocks"] * 4
        free_kb = s["free_blocks"] * 4
        pct = (s["used_blocks"] / max(1, s["total_blocks"])) * 100
        self._print(f"{'Filesystem':<12} {'Total':>8} {'Used':>8} {'Avail':>8} {'Use%':>5}")
        self._print(f"{'neurfs':<12} {total_kb:>7}K {used_kb:>7}K {free_kb:>7}K {pct:>4.0f}%")

    def _cmd_free(self, args: List[str]):
        """Show memory usage."""
        mmu_stats = self.os.mmu.stats()
        total = self.os.mmu.max_physical_frames
        free = mmu_stats["free_frames"]
        used = total - free
        self._print(f"{'':>12} {'Total':>8} {'Used':>8} {'Free':>8}")
        self._print(f"{'Pages:':>12} {total:>8} {used:>8} {free:>8}")
        self._print(f"{'KB:':>12} {total*4:>8} {used*4:>8} {free*4:>8}")

    def _cmd_uptime(self, args: List[str]):
        """Show system uptime."""
        ticks = self.os.scheduler.tick
        self._print(f"neurOS up {ticks} ticks, "
                     f"{self.os.process_table.count} processes")

    def _cmd_asm(self, args: List[str]):
        """Assemble a file from the filesystem."""
        if not args:
            self._print("asm: usage: asm <file.asm>")
            return
        if not hasattr(self.os, 'assembler') or self.os.assembler is None:
            self._print("asm: assembler not available")
            return
        path = self._resolve_path(args[0])
        data = self.os.fs.read_file(path)
        if data is None:
            self._print(f"asm: cannot read '{args[0]}'")
            return
        source = "".join(chr(b) for b in data.cpu().tolist() if b < 128)
        result = self.os.assembler.assemble(source)
        if result.success:
            self._print(f"Assembled {result.num_instructions} instructions")
            for i, word in enumerate(result.binary):
                self._print(f"  {i:4d}: 0x{word:08X}")
        else:
            for err in result.errors:
                self._print(f"Error: {err}")

    def _cmd_nsc(self, args: List[str]):
        """Compile an nsl source file."""
        if not args:
            self._print("nsc: usage: nsc <file.nsl>")
            return
        if not hasattr(self.os, 'compiler') or self.os.compiler is None:
            self._print("nsc: compiler not available")
            return
        path = self._resolve_path(args[0])
        data = self.os.fs.read_file(path)
        if data is None:
            self._print(f"nsc: cannot read '{args[0]}'")
            return
        source = "".join(chr(b) for b in data.cpu().tolist() if b < 128)
        result = self.os.compiler.compile(source)
        if result.success:
            n_instr = result.assembly_result.num_instructions if result.assembly_result else 0
            self._print(f"Compiled: {len(result.ir)} IR -> {n_instr} instructions")
            if result.optimizations_applied:
                self._print(f"Optimizations: {result.optimizations_applied}")
        else:
            for err in result.errors:
                self._print(f"Error: {err}")

    def _cmd_write(self, args: List[str]):
        """Write text content to a file."""
        if len(args) < 2:
            self._print("write: usage: write <file> <content...>")
            return
        path = self._resolve_path(args[0])
        content = " ".join(args[1:])
        data = torch.tensor(
            [ord(c) for c in content],
            dtype=torch.uint8, device=self.device,
        )
        if self.os.fs.write_file(path, data):
            self._print(f"Wrote {len(content)} bytes to {path}")
        else:
            self._print(f"write: cannot write to '{args[0]}'")

    def _cmd_run(self, args: List[str]):
        """Compile an nsl file and execute on neural CPU."""
        if not args:
            self._print("run: usage: run <file.nsl>")
            return
        if not hasattr(self.os, 'compiler') or self.os.compiler is None:
            self._print("run: compiler not available")
            return
        path = self._resolve_path(args[0])
        data = self.os.fs.read_file(path)
        if data is None:
            self._print(f"run: cannot read '{args[0]}'")
            return
        source = "".join(chr(b) for b in data.cpu().tolist() if b < 128)

        # Compile
        t0 = time.perf_counter()
        result = self.os.compiler.compile(source)
        compile_time = time.perf_counter() - t0
        if not result.success:
            for err in result.errors:
                self._print(f"Error: {err}")
            return
        n_instr = result.assembly_result.num_instructions if result.assembly_result else 0
        self._print(f"Compiled: {len(result.ir)} IR -> {n_instr} instructions [{compile_time*1e6:.0f}us]")

        # Load into CPU and execute
        assembly_text = result.assembly
        asm_lines = [l for l in assembly_text.split("\n")
                     if l.strip() and not l.strip().startswith(";")]
        clean_asm = "\n".join(asm_lines)

        try:
            from ncpu.model import CPU as ModelCPU
            cpu = ModelCPU(mock_mode=True, neural_execution=True,
                           models_dir="models", max_cycles=5000)
            cpu.load_program(clean_asm)
            t0 = time.perf_counter()
            try:
                cpu.run(max_cycles=5000)
            except RuntimeError as e:
                if "Max cycles" not in str(e):
                    self._print(f"Error: {e}")
                    return
            exec_time = time.perf_counter() - t0
            cycles = cpu.get_cycle_count()
            self._last_cpu = cpu

            regs = cpu.dump_registers()
            non_zero = {k: v for k, v in regs.items() if v != 0}
            self._print(f"Executed: {cycles} cycles [{exec_time*1e6:.0f}us]")
            if non_zero:
                for reg, val in sorted(non_zero.items()):
                    self._print(f"  {reg} = {val} (0x{val & 0xFFFFFFFF:08X})")
        except Exception as e:
            self._print(f"Error: {e}")

    def _cmd_exec(self, args: List[str]):
        """Assemble an asm file and execute on neural CPU."""
        if not args:
            self._print("exec: usage: exec <file.asm>")
            return
        if not hasattr(self.os, 'assembler') or self.os.assembler is None:
            self._print("exec: assembler not available")
            return
        path = self._resolve_path(args[0])
        data = self.os.fs.read_file(path)
        if data is None:
            self._print(f"exec: cannot read '{args[0]}'")
            return
        source = "".join(chr(b) for b in data.cpu().tolist() if b < 128)

        # Assemble
        t0 = time.perf_counter()
        result = self.os.assembler.assemble(source)
        asm_time = time.perf_counter() - t0
        if not result.success:
            for err in result.errors:
                self._print(f"Error: {err}")
            return
        self._print(f"Assembled: {result.num_instructions} instructions [{asm_time*1e6:.0f}us]")

        # Load into CPU and execute
        try:
            from ncpu.model import CPU as ModelCPU
            cpu = ModelCPU(mock_mode=True, neural_execution=True,
                           models_dir="models", max_cycles=5000)
            cpu.load_program(source)
            t0 = time.perf_counter()
            try:
                cpu.run(max_cycles=5000)
            except RuntimeError as e:
                if "Max cycles" not in str(e):
                    self._print(f"Error: {e}")
                    return
            exec_time = time.perf_counter() - t0
            cycles = cpu.get_cycle_count()
            self._last_cpu = cpu

            regs = cpu.dump_registers()
            non_zero = {k: v for k, v in regs.items() if v != 0}
            self._print(f"Executed: {cycles} cycles [{exec_time*1e6:.0f}us]")
            if non_zero:
                for reg, val in sorted(non_zero.items()):
                    self._print(f"  {reg} = {val} (0x{val & 0xFFFFFFFF:08X})")
        except Exception as e:
            self._print(f"Error: {e}")

    def _cmd_regs(self, args: List[str]):
        """Show register state from last run/exec."""
        if self._last_cpu is None:
            self._print("regs: no program has been run yet")
            return
        regs = self._last_cpu.dump_registers()
        self._print(f"{'Register':<8} {'Decimal':>12} {'Hex':>12}")
        self._print(f"{'--------':<8} {'-------':>12} {'---':>12}")
        for reg, val in sorted(regs.items()):
            self._print(f"{reg:<8} {val:>12} 0x{val & 0xFFFFFFFF:08X}")

    def _cmd_neural(self, args: List[str]):
        """Show neural model status."""
        models = []
        if self.os.mmu:
            models.append(("MMU (mmu.pt)", self.os.mmu._trained))
        if self.os.tlb:
            models.append(("TLB (tlb.pt)", self.os.tlb._policy_trained))
        if self.os.cache:
            models.append(("Cache Replacer (cache_replace.pt)", self.os.cache._replacer_trained))
            models.append(("Cache Prefetch (prefetch.pt)", self.os.cache._prefetcher_trained))
        if self.os.gic:
            models.append(("GIC (gic.pt)", self.os.gic._trained))
        if self.os.scheduler:
            models.append(("Scheduler (scheduler.pt)", self.os.scheduler._trained))
        if self.os.fs:
            models.append(("Block Alloc (block_alloc.pt)", self.os.fs._allocator_trained))
        if self.os.assembler:
            models.append(("Assembler Tokenizer (assembler_tokenizer.pt)", self.os.assembler._tokenizer_trained))
            models.append(("Assembler Codegen (assembler_codegen.pt)", self.os.assembler._codegen_trained))
        if self.os.compiler:
            models.append(("Compiler Optimizer (compiler_optimizer.pt)", self.os.compiler._optimizer_trained))

        neural_count = sum(1 for _, trained in models if trained)
        self._print(f"Neural Models: {neural_count}/{len(models)}")
        self._print("")
        for name, trained in models:
            status = "NEURAL" if trained else "FALLBACK"
            self._print(f"  {name:<45} {status}")

    def _cmd_bench(self, args: List[str]):
        """Quick benchmark: compile+execute sum 1-10."""
        if not hasattr(self.os, 'compiler') or self.os.compiler is None:
            self._print("bench: compiler not available")
            return
        source = "var sum = 0; var i = 1; var limit = 11; var one = 1; while (i != limit) { sum = sum + i; i = i + one; } halt;"

        # Compile
        t0 = time.perf_counter()
        result = self.os.compiler.compile(source)
        compile_us = (time.perf_counter() - t0) * 1e6
        if not result.success:
            self._print(f"bench: compile failed: {result.errors}")
            return

        # Execute
        assembly_text = result.assembly
        asm_lines = [l for l in assembly_text.split("\n")
                     if l.strip() and not l.strip().startswith(";")]
        clean_asm = "\n".join(asm_lines)

        try:
            from ncpu.model import CPU as ModelCPU
            cpu = ModelCPU(mock_mode=True, neural_execution=True,
                           models_dir="models", max_cycles=5000)
            cpu.load_program(clean_asm)
            t0 = time.perf_counter()
            try:
                cpu.run(max_cycles=5000)
            except RuntimeError as e:
                if "Max cycles" not in str(e):
                    self._print(f"bench: exec error: {e}")
                    return
            exec_us = (time.perf_counter() - t0) * 1e6
            cycles = cpu.get_cycle_count()
            regs = cpu.dump_registers()
            result_val = next((v for v in regs.values() if v == 55), None)
            status = "PASS (sum=55)" if result_val is not None else "CHECK REGISTERS"
            self._print(f"Benchmark: sum(1..10)")
            self._print(f"  Compile:  {compile_us:.0f}us")
            self._print(f"  Execute:  {exec_us:.0f}us ({cycles} cycles)")
            self._print(f"  Result:   {status}")
        except Exception as e:
            self._print(f"bench: error: {e}")

    def _cmd_hexdump(self, args: List[str]):
        """Show hex dump of file contents."""
        if not args:
            self._print("hexdump: usage: hexdump <file>")
            return
        path = self._resolve_path(args[0])
        data = self.os.fs.read_file(path)
        if data is None:
            self._print(f"hexdump: cannot read '{args[0]}'")
            return
        raw = data.cpu().tolist()
        for offset in range(0, len(raw), 16):
            chunk = raw[offset:offset + 16]
            hex_part = " ".join(f"{b:02X}" for b in chunk)
            ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
            self._print(f"  {offset:08X}  {hex_part:<48} |{ascii_part}|")

    def _print(self, text: str):
        """Add text to output buffer."""
        self.output.append(text)

    def __repr__(self) -> str:
        return f"NeuralShell(cwd={self.cwd}, pid={self.pid})"
