from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.console import Console
from rich.align import Align
from rich.text import Text

from collections import deque
from statistics import mean
import os
import yaml

def generate_table(rows):
    layout = Layout()
    console = Console()

    table = Table(show_header=True, header_style="bold magenta", expand=True)
    table.add_column("Time", vertical="middle")
    table.add_column("Cycle", justify="center", vertical="middle")
    table.add_column(Text("L1A Rate [kHz]", overflow="fold"), justify="right", vertical="middle")
    table.add_column("FIFO Occupancy", justify="right", vertical="middle")
    table.add_column("RB RT1 Temp [°C]", justify="right", vertical="middle")
    table.add_column("RB RT2 Temp [°C]", justify="right", vertical="middle")
    table.add_column("RB SCA Temp [°C]", justify="right", vertical="middle")
    table.add_column("RB VTRX Temp [°C]", justify="right", vertical="middle")
    table.add_column("FIFO Lost", justify="right", vertical="middle")
    table.add_column("Packet Rate [kHz]", justify="right", vertical="middle")

    rows = list(rows)

    # Get the height:
    render_map = layout.render(console, console.options)
    n_rows = render_map[layout].region.height

    while n_rows >= 0:
        table = Table(show_header=True, header_style="bold magenta", expand=True)
        table.add_column("Time", vertical="middle")
        table.add_column("Cycle", justify="center", vertical="middle")
        table.add_column(Text("L1A Rate [kHz]", overflow="fold"), justify="right", vertical="middle")
        table.add_column("FIFO Occupancy", justify="right", vertical="middle")
        table.add_column("RB RT1 Temp [°C]", justify="right", vertical="middle")
        table.add_column("RB RT2 Temp [°C]", justify="right", vertical="middle")
        table.add_column("RB SCA Temp [°C]", justify="right", vertical="middle")
        table.add_column("RB VTRX Temp [°C]", justify="right", vertical="middle")
        table.add_column("FIFO Lost", justify="right", vertical="middle")
        table.add_column("Packet Rate [kHz]", justify="right", vertical="middle")

        for row in rows[-n_rows:]:
            table.add_row(*row)

        layout.update(table)

        render_map = layout.render(console, console.options)

        if len(render_map[layout].render[-1]) > 2:
            # The table is overflowing
            n_rows -= 1
        else:
            break

    return table

def generate_header() -> Panel:
    tamalero_header = """
    ████████╗ █████╗ ███╗   ███╗ █████╗ ██╗     ███████╗███████╗\n\
    ╚══██╔══╝██╔══██╗████╗ ████║██╔══██╗██║     ██╔════╝██╔════╝\n\
       ██║   ███████║██╔████╔██║███████║██║     █████╗  ███████╗\n\
       ██║   ██╔══██║██║╚██╔╝██║██╔══██║██║     ██╔══╝  ╚════██║\n\
       ██║   ██║  ██║██║ ╚═╝ ██║██║  ██║███████╗███████╗███████║\n\
       ╚═╝   ╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝\n\
    """.splitlines()

    grid = Table.grid(expand=True)
    grid.add_column(justify="center", ratio=1)
    for i, line in enumerate(tamalero_header):
        grid.add_row(tamalero_header[i])

    grid.add_row("[underline italic]Fermilab Test Beam Simulation")

    return Panel(grid, style="magenta")

def generate_static(l1as, l1a_rate, nmin) -> Panel:

    grid = Table.grid(expand=True)
    grid.add_column(justify="left", ratio=1)
    grid.add_column(justify="left", ratio=2)

    grid.add_row("[bold]Current Directory:", Text(os.getcwd(), overflow="fold"))
    grid.add_row("[bold]Input Parameters:", f"l1a_rate = {l1a_rate} kHz  ;  nmin = {nmin}")

    for cycle, rates in l1as.items():
        avg_l1as = mean(rates) if len(rates) > 0 else 0
        grid.add_row("[bold]Average L1A Rate:", f"{avg_l1as:.2f} kHz\tCycle: {cycle}")

    return Panel(grid)

def generate_files(beam) -> Panel:
    table = Table(expand=True, show_lines=True)
    table.add_column("File", justify="center")
    table.add_column("Written?", justify="center")
    table.add_column("Zipped?", justify="center")
    table.add_column("ETROC?", justify="center")

    if beam.files:
        for (filename, zipped) in beam.files.items():
            if zipped:
                table.add_row(Text(filename, overflow="fold"), ":white_check_mark:", ":white_check_mark:", ":white_check_mark:")
            else:
                table.add_row(Text(filename, overflow="fold"), ":white_check_mark:", "...", "...")

    return Panel(Align.center(table))

def read_adrs(addresses: list) -> list:
    adr = []
    for a in addresses:
        tmp_adr = read_adr(a)
        adr.append(tmp_adr)
    return adr

def read_adr(address: str) -> int:
    tmp_adr = ""
    for ch in address[::-1]:
        try:
            int(ch)
            tmp_adr += ch
        except ValueError:
            break
    tmp_adr = int(tmp_adr[::-1])
    return tmp_adr

def read_etroc(beam, time_stamp, full=False):
    """
    Reads ETROC registers and dumps them to a yaml file. Currently only implemented for peripheral registers (full=False)
    """
    etrocs = {"ETROC2":{}}
    if full:
        for register in beam.module.regs:
            etrocs["ETROC2"][register] = {}
            try:
                mask = beam.module.regs[register]['mask']
                shift = beam.module.regs[register]['shift']
                regadr = beam.module.regs[register]['regadr']
                if type(regadr) == dict:
                    for key, val in regadr.items():
                        if type(val) == list:
                            adr = read_adrs(val)
                            value = [(beam.module.I2C_read(a)&m) >> s for a, m, s in zip(adr, mask, shift)]
                            for v, a in zip(value, adr):
                                etrocs["ETROC2"][register][v] = a
                        elif type(val) == str:
                            adr = read_adr(val)
                            value = (beam.module.I2C_read(adr)&mask) >> shift
                            etrocs["ETROC2"][register][value] = adr
                elif type(regadr) == list:
                    adr = read_adrs(regadr)
                    value = [(beam.module.I2C_read(a)&m) >> s for a, m, s in zip(adr, mask, shift)]
                    for v, a in zip(value, adr):
                        etrocs["ETROC2"][register][v] = a
                elif type(regadr) == str:
                    adr = read_adr(regadr)
                    value = (beam.module.I2C_read(adr)&mask) >> shift
                    etrocs["ETROC2"][register][value] = adr
            except:
                pass
    else:
        for register in beam.module.regs:
            etrocs["ETROC2"][register] = {}
            try:
                value = beam.module.rd_reg(register)
                adr = int(beam.module.regs[register]['regadr'])
                etrocs["ETROC2"][register][hex(adr)] = value
            except:
                pass
    
    with open(f"output/read_beam_{beam.START}/etroc_{time_stamp}.yaml", mode="w") as f:
        yaml.dump(etrocs, f)
