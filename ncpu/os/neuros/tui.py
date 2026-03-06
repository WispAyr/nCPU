"""neurOS TUI — Interactive Neural Operating System Terminal.

Rich-formatted output, prompt_toolkit shell loop with tab completion,
animated boot sequence, and a proper terminal experience.

Usage:
    python ncpu/os/tui.py
    python -m ncpu.os.tui
    python ncpu/os/tui.py --no-animation
    python ncpu/os/tui.py --device cpu
"""

import sys
import time
import argparse
import logging

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.markup import escape

from prompt_toolkit import PromptSession, HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.completion import Completer, Completion

sys.path.insert(0, ".")
from ncpu.os import NeurOS

logging.basicConfig(level=logging.WARNING)

# Commands that take file path arguments
_FILE_CMDS = {"cat", "rm", "stat", "asm", "nsc", "run", "exec", "write", "hexdump", "touch"}
_DIR_CMDS = {"cd", "ls", "mkdir", "rmdir"}


class NeurOSCompleter(Completer):
    """Tab completion for neurOS shell commands and filesystem paths."""

    def __init__(self, nos: NeurOS):
        self.nos = nos

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        parts = text.split()

        if len(parts) == 0 or (len(parts) == 1 and not text.endswith(" ")):
            # Complete command name
            prefix = parts[0] if parts else ""
            for cmd in sorted(self.nos.shell._builtins.keys()):
                if cmd.startswith(prefix):
                    yield Completion(cmd, start_position=-len(prefix))
            return

        # Complete file/directory paths for relevant commands
        cmd = parts[0]
        if cmd not in _FILE_CMDS and cmd not in _DIR_CMDS:
            return

        # Get the partial path being typed
        partial = parts[-1] if text.endswith(" ") is False and len(parts) > 1 else ""
        if text.endswith(" "):
            partial = ""

        # Resolve directory to list
        shell = self.nos.shell
        if "/" in partial:
            # Completing within a subdirectory
            last_slash = partial.rfind("/")
            dir_part = partial[:last_slash] if last_slash > 0 else "/"
            name_prefix = partial[last_slash + 1:]
            search_dir = shell._resolve_path(dir_part)
            path_prefix = partial[:last_slash + 1]
        else:
            search_dir = shell.cwd
            name_prefix = partial
            path_prefix = ""

        entries = self.nos.fs.list_dir(search_dir)
        if entries is None:
            return

        dirs_only = cmd in _DIR_CMDS

        for name in sorted(entries):
            if not name.startswith(name_prefix):
                continue
            entry_path = f"{search_dir}/{name}" if search_dir != "/" else f"/{name}"
            info = self.nos.fs.stat(entry_path)
            is_dir = info and info["type"] == "DIRECTORY"

            if dirs_only and not is_dir:
                continue

            display = f"{name}/" if is_dir else name
            completion = f"{path_prefix}{display}"
            yield Completion(completion, start_position=-len(partial))


class NeurOSTUI:
    """Interactive TUI for neurOS."""

    def __init__(self, device=None):
        self.console = Console()
        self.nos = NeurOS(device=device)
        self.boot_stages = {}
        self.models_loaded = 0

    def boot(self, animate=True):
        """Boot neurOS with Rich-formatted output."""
        # Banner
        banner = Text()
        banner.append("neurOS", style="bold cyan")
        banner.append(" v0.1 — GPU-Native Neural Operating System\n", style="white")
        banner.append(f"Device: {self.nos.device}", style="dim")
        self.console.print(Panel(banner, border_style="cyan", padding=(0, 2)))

        if animate:
            self.console.print()

        # Boot with timing capture
        t_start = time.perf_counter()
        self.boot_stages = self.nos.boot(quiet=True)
        total_ms = (time.perf_counter() - t_start) * 1000

        # Count loaded models
        self.models_loaded = self._count_neural_models()

        if animate:
            self._print_boot_stages()
        else:
            self.console.print(f"[dim]Boot complete in {total_ms:.0f}ms[/dim]")

        self.console.print()
        summary = Text()
        summary.append(f"neurOS ready", style="bold green")
        summary.append(f" — {self.models_loaded} neural models loaded", style="green")
        summary.append(f" — {total_ms:.0f}ms boot", style="dim")
        self.console.print(summary)
        self.console.print("[dim]Type 'help' for available commands. Ctrl+D to exit.[/dim]")
        self.console.print()

    def _count_neural_models(self) -> int:
        count = 0
        nos = self.nos
        if nos.mmu and nos.mmu._trained:
            count += 1
        if nos.tlb and nos.tlb._policy_trained:
            count += 1
        if nos.cache and nos.cache._replacer_trained:
            count += 1
        if nos.cache and nos.cache._prefetcher_trained:
            count += 1
        if nos.gic and nos.gic._trained:
            count += 1
        if nos.scheduler and nos.scheduler._trained:
            count += 1
        if nos.fs and nos.fs._allocator_trained:
            count += 1
        if nos.assembler and nos.assembler._tokenizer_trained:
            count += 1
        if nos.assembler and nos.assembler._codegen_trained:
            count += 1
        if nos.compiler and nos.compiler._optimizer_trained:
            count += 1
        return count

    def _print_boot_stages(self):
        """Print boot stages with timing and checkmarks."""
        stage_labels = [
            ("memory", "Memory subsystem"),
            ("interrupts", "Interrupt controller"),
            ("processes", "Process management"),
            ("ipc", "IPC subsystem"),
            ("filesystem", "Filesystem"),
            ("shell", "Shell"),
            ("toolchain", "Assembler & Compiler"),
            ("models", f"Neural models ({self.models_loaded} loaded)"),
        ]

        for key, label in stage_labels:
            ms = self.boot_stages.get(key, 0) * 1000
            dots = "." * (40 - len(label))
            line = Text()
            line.append("[BOOT] ", style="bold cyan")
            line.append(f"{label} {dots} ", style="white")
            line.append(f"{ms:6.1f}ms", style="yellow")
            line.append("  ✓", style="bold green")
            self.console.print(line)
            time.sleep(0.04)  # Slight animation delay

    def run_shell(self):
        """Run the interactive shell loop with prompt_toolkit."""
        session = PromptSession(
            history=InMemoryHistory(),
            completer=NeurOSCompleter(self.nos),
            complete_while_typing=False,
        )

        shell = self.nos.shell

        while shell.running:
            try:
                cwd = shell.cwd
                prompt = HTML(
                    f'<ansigreen>root@neuros</ansigreen>'
                    f':<ansiblue>{escape(cwd)}</ansiblue># '
                )
                cmd = session.prompt(
                    prompt,
                    bottom_toolbar=self._get_toolbar,
                )

                if not cmd.strip():
                    continue

                if cmd.strip() == "clear":
                    self.console.clear()
                    continue

                output = shell.execute(cmd)
                self._format_output(cmd.strip(), output)

            except KeyboardInterrupt:
                self.console.print()  # newline after ^C
                continue
            except EOFError:
                self.console.print("\n[dim]Goodbye.[/dim]")
                break

    def _get_toolbar(self):
        boot_ms = self.boot_stages.get("total", 0) * 1000
        return HTML(
            f'<b>neurOS</b> | Device: {escape(str(self.nos.device))} '
            f'| Models: {self.models_loaded}/10 '
            f'| Boot: {boot_ms:.0f}ms'
        )

    def _format_output(self, cmd: str, output: list):
        """Route output formatting based on command name."""
        if not output:
            return

        cmd_name = cmd.split()[0] if cmd else ""

        if cmd_name == "ls":
            self._format_ls(output)
        elif cmd_name == "ps":
            self._format_ps(output)
        elif cmd_name == "top":
            self._format_top(output)
        elif cmd_name == "df":
            self._format_df(output)
        elif cmd_name == "free":
            self._format_free(output)
        elif cmd_name == "help":
            self._format_help(output)
        elif cmd_name == "neural":
            self._format_neural(output)
        elif cmd_name == "regs":
            self._format_regs(output)
        elif cmd_name in ("asm", "nsc"):
            self._format_asm(output)
        elif cmd_name in ("run", "exec"):
            self._format_run(output)
        elif cmd_name == "uname":
            for line in output:
                self.console.print(f"[bold green]{escape(line)}[/bold green]")
        else:
            self._format_default(output)

    def _format_ls(self, output):
        for line in output:
            text = line.strip()
            if text.endswith("/"):
                self.console.print(f"  [bold blue]{escape(text)}[/bold blue]")
            elif "No such" in text or "cannot" in text:
                self.console.print(f"[bold red]{escape(line)}[/bold red]")
            else:
                # file with size
                parts = text.rsplit("(", 1)
                if len(parts) == 2:
                    name = parts[0].strip()
                    size = f"({parts[1]}"
                    self.console.print(f"  {escape(name)}  [dim]{escape(size)}[/dim]")
                else:
                    self.console.print(f"  {escape(text)}")

    def _format_ps(self, output):
        if len(output) < 2:
            for line in output:
                self.console.print(escape(line))
            return

        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
        table.add_column("PID", justify="right", style="cyan")
        table.add_column("STATE", justify="left")
        table.add_column("PRI", justify="right")
        table.add_column("CPU", justify="right")
        table.add_column("NAME")

        state_colors = {
            "RUNNING": "green",
            "READY": "yellow",
            "BLOCKED": "yellow",
            "SLEEPING": "blue",
            "ZOMBIE": "red",
            "TERMINATED": "red",
        }

        for line in output[2:]:  # Skip headers
            parts = line.split()
            if len(parts) >= 5:
                pid, state, pri, cpu_t = parts[0], parts[1], parts[2], parts[3]
                name = " ".join(parts[4:])
                color = state_colors.get(state, "white")
                table.add_row(pid, f"[{color}]{state}[/{color}]", pri, cpu_t, name)

        self.console.print(table)

    def _format_top(self, output):
        text = "\n".join(output)
        self.console.print(Panel(escape(text), title="[bold]System Status[/bold]",
                                 border_style="cyan"))

    def _format_df(self, output):
        if len(output) < 2:
            for line in output:
                self.console.print(escape(line))
            return

        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
        table.add_column("Filesystem")
        table.add_column("Total", justify="right")
        table.add_column("Used", justify="right")
        table.add_column("Avail", justify="right")
        table.add_column("Use%", justify="right")
        table.add_column("Bar")

        for line in output[1:]:
            parts = line.split()
            if len(parts) >= 5:
                pct_str = parts[4].rstrip("%")
                try:
                    pct = float(pct_str)
                except ValueError:
                    pct = 0
                filled = int(pct / 5)
                bar = f"[green]{'█' * filled}[/green][dim]{'░' * (20 - filled)}[/dim]"
                table.add_row(parts[0], parts[1], parts[2], parts[3], parts[4], bar)

        self.console.print(table)

    def _format_free(self, output):
        if len(output) < 2:
            for line in output:
                self.console.print(escape(line))
            return

        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
        table.add_column("", justify="right")
        table.add_column("Total", justify="right", style="cyan")
        table.add_column("Used", justify="right", style="yellow")
        table.add_column("Free", justify="right", style="green")

        for line in output[1:]:
            parts = line.split()
            if len(parts) >= 4:
                table.add_row(parts[0], parts[1], parts[2], parts[3])

        self.console.print(table)

    def _format_help(self, output):
        if not output:
            return
        # First line is the header
        self.console.print(f"[bold]{escape(output[0])}[/bold]")
        if len(output) > 1 and output[1].strip() == "":
            pass  # skip blank line

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Command", style="cyan", min_width=22)
        table.add_column("Description")

        for line in output[2:] if len(output) > 2 else output[1:]:
            stripped = line.strip()
            if not stripped:
                continue
            # Parse "  cmd<spaces>description" format
            parts = stripped.split(None, 1)
            # Handle multi-word commands like "ls [path]"
            # The format is "  cmd_with_args<20 spaces>description"
            # Let's just split on double-space
            idx = stripped.find("  ")
            if idx > 0:
                cmd_part = stripped[:idx].strip()
                desc_part = stripped[idx:].strip()
                table.add_row(cmd_part, desc_part)
            elif len(parts) >= 2:
                table.add_row(parts[0], parts[1])
            else:
                table.add_row(stripped, "")

        self.console.print(table)

    def _format_neural(self, output):
        if not output:
            return

        # First line is summary
        self.console.print(f"[bold]{escape(output[0])}[/bold]")

        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
        table.add_column("Model", min_width=40)
        table.add_column("Status", justify="center")

        for line in output[2:]:  # Skip summary and blank line
            stripped = line.strip()
            if not stripped:
                continue
            # Parse "  name<spaces>STATUS" format
            if "NEURAL" in stripped:
                idx = stripped.rfind("NEURAL")
                name = stripped[:idx].strip()
                table.add_row(name, "[bold green]NEURAL[/bold green]")
            elif "FALLBACK" in stripped:
                idx = stripped.rfind("FALLBACK")
                name = stripped[:idx].strip()
                table.add_row(name, "[bold red]FALLBACK[/bold red]")

        self.console.print(table)

    def _format_regs(self, output):
        if not output:
            return

        if "no program" in output[0]:
            self.console.print(f"[yellow]{escape(output[0])}[/yellow]")
            return

        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
        table.add_column("Register", style="cyan")
        table.add_column("Decimal", justify="right")
        table.add_column("Hex", justify="right", style="dim")

        for line in output[2:]:  # Skip header rows
            parts = line.split()
            if len(parts) >= 3:
                reg = parts[0]
                dec = parts[1]
                hexval = parts[2]
                style = "bold" if dec != "0" else "dim"
                table.add_row(f"[{style}]{reg}[/{style}]", f"[{style}]{dec}[/{style}]",
                              hexval)

        self.console.print(table)

    def _format_asm(self, output):
        for line in output:
            if line.startswith("Assembled") or line.startswith("Compiled"):
                self.console.print(f"[green]{escape(line)}[/green]")
            elif line.startswith("Error:"):
                self.console.print(f"[bold red]{escape(line)}[/bold red]")
            elif line.strip().startswith("0x") or ": 0x" in line:
                self.console.print(f"[cyan]{escape(line)}[/cyan]")
            else:
                self.console.print(escape(line))

    def _format_run(self, output):
        for line in output:
            if line.startswith("Compiled") or line.startswith("Assembled"):
                self.console.print(f"[green]{escape(line)}[/green]")
            elif line.startswith("Executed"):
                self.console.print(f"[green]{escape(line)}[/green]")
            elif line.startswith("Error:"):
                self.console.print(f"[bold red]{escape(line)}[/bold red]")
            elif line.strip().startswith("R") and "=" in line:
                self.console.print(f"[cyan]{escape(line)}[/cyan]")
            elif line.startswith("Benchmark") or line.startswith("  Compile") or line.startswith("  Execute") or line.startswith("  Result"):
                self.console.print(f"[cyan]{escape(line)}[/cyan]")
            else:
                self.console.print(escape(line))

    def _format_default(self, output):
        for line in output:
            if any(kw in line for kw in ("Error:", "cannot", "No such", "not found")):
                self.console.print(f"[bold red]{escape(line)}[/bold red]")
            else:
                self.console.print(escape(line))


def main():
    parser = argparse.ArgumentParser(description="neurOS Terminal")
    parser.add_argument("--device", default=None, help="Compute device (cpu, cuda, mps)")
    parser.add_argument("--no-animation", action="store_true", help="Skip boot animation")
    args = parser.parse_args()

    import torch
    device = torch.device(args.device) if args.device else None

    tui = NeurOSTUI(device=device)
    tui.boot(animate=not args.no_animation)
    tui.run_shell()


if __name__ == "__main__":
    main()
