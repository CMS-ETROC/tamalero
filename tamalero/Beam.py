#!/usr/bin/env python3
import uhal
import os
import time
import struct
import datetime
from tamalero.Module import Module
from threading import Thread

class Beam():
    def __init__(self, rb):
        try:
            self.rb  = rb
            self.kcu = self.rb.kcu
            self.module = Module(self.rb)
        except:
            print("Unable to connect to KCU.")
        self.ON = 4
        self.OFF = 56
        self.cycles = 0
        self.start_timer = False
        self.dashboard = False

    def generate_beam(self, l1a_rate, nmin, verbose=False):
        """
        Simulates Fermilab's test beam (4s ON, 56s OFF), sending L1A signals at l1a_rate kHz [default = 1000] for nmin minutes [default = 1]
        """

        self.SIM = True

        if verbose: print("Preparing beam...")

        uhal.disableLogging()

        self.l1a_rate = l1a_rate
        self.nmin = nmin

        self.trigger_rate = self.l1a_rate * 1000 / 25E-9 / (0xffffffff) * 10000

        ON_TIME = self.ON
        OFF_TIME = self.OFF

        self.START = ""
        if verbose: verbose_start = time.time()

        for minute in range(self.nmin):

            writer_ON = Thread(target=self.kcu.write_node, args=("SYSTEM.L1A_RATE", int(self.trigger_rate)))
            sleeper_ON = Thread(target=time.sleep, args=(ON_TIME,))

            while not self.start_timer:
                if not self.dashboard: break
                continue

            if verbose:
                print("### Beam ON ###")
                start_ON = time.time()
                verbose_start = start_ON
            if minute == 0: self.START = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

            writer_ON.start()
            sleeper_ON.start()

            writer_ON.join()
            sleeper_ON.join()

            if verbose:
                print("Shutting off beam...")
                print("\tON time  = {:.2f} s".format(time.time() - start_ON))
                print("\tL1A rate = {:.2f} MHz".format(self.kcu.read_node("SYSTEM.L1A_RATE_CNT").value()/1000000.0))

                print("### Beam OFF ###")
                start_OFF = time.time()

            writer_OFF = Thread(target=self.kcu.write_node, args=("SYSTEM.L1A_RATE", 0))
            sleeper_OFF = Thread(target=time.sleep, args=(OFF_TIME,))

            writer_OFF.start()
            sleeper_OFF.start()

            writer_OFF.join()
            sleeper_OFF.join()

            self.cycles += 1

            if verbose:
                print("{} minutes completed".format(minute+1))
                print("\tOFF time = {:.2f} s".format(time.time() - start_OFF))
                print("\tL1A rate = {:.2f} MHz".format(self.kcu.read_node("SYSTEM.L1A_RATE_CNT").value()/1000000.0))

        if verbose:
            total_time = round(time.time() - verbose_start)
            total_time = str(datetime.timedelta(seconds=total_time))
            print("Test beam simulation completed; it took {}.".format(total_time))

        self.SIM = False

    def read_fifo(self, block=255, verbose=False):
        # from tamalero.beam_utils import read_etroc
        while not self.start_timer:
            if not self.dashboard: break
            continue
        self.files = {}
        while self.SIM:
            data = []
            while self.kcu.read_node("SYSTEM.L1A_RATE_CNT") != 0 or self.kcu.read_node("READOUT_BOARD_0.RX_FIFO_OCCUPANCY") != 0:
                try:
                    # Check FIFO occupancy
                    occupancy = self.kcu.read_node("READOUT_BOARD_0.RX_FIFO_OCCUPANCY")
                    num_blocks_to_read = occupancy.value() // block

                    # Read data from FIFO
                    if (num_blocks_to_read):
                        reads = num_blocks_to_read * [self.kcu.hw.getNode("DAQ_RB0").readBlock(block)]  # reads is a list of num_blocks_to_read elements of type uhal._core.ValVector_uint32
                        self.kcu.dispatch()
                        for read in reads:
                            data += read.value()
                    elif (not num_blocks_to_read) and (occupancy.value() > 0):
                        reads = self.kcu.hw.getNode("DAQ_RB0").readBlock(occupancy.value())   # reads is a single uhal._core.ValVector_uint32 element, not a list; not iterable
                        self.kcu.dispatch()
                        data += reads.value()

                except uhal._core.exception:
                    print("uhal UDP error in daq")

            if data:
                time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                if not os.path.isdir(f"output/read_beam_{self.START}"): os.system(f"mkdir output/read_beam_{self.START}")
                filename = f"output/read_beam_{self.START}/read_beam_{time_stamp}.dat"
                self.files[filename] = False
                with open(filename, mode="wb") as f:
                    f.write(struct.pack('<{}I'.format(len(data)), *data))
                # read_etroc(self, time_stamp, full=False)
                os.system(f"gzip {filename}")
                self.files[filename] = True

    def monitor(self):
        from rich.live import Live
        from rich.layout import Layout
        from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, TextColumn
        from rich.panel import Panel
        from rich.align import Align

        from tamalero.beam_utils import generate_table, generate_header, generate_static, generate_files

        from collections import deque
        import os

        self.dashboard = True

        layout = Layout(name="root")

        layout.split_column(
            Layout(name="header", size=12),
            Layout(name="main", ratio=1)
        )

        layout["main"].split_row(
            Layout(name="dynamic", ratio=3),
            Layout(name="parameters", ratio=1),
        )
        layout["parameters"].split_column(
            Layout(name="static"),
            Layout(name="files"),
            Layout(name="progress", size=5)
       )

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            TextColumn(f"/ 0:{self.nmin:02d}:00")
        )
        total_task = self.nmin * 60
        sim_task = progress.add_task("[bold green]Beam Simulation", total=total_task, start=False)

        width, height = os.get_terminal_size()
        table_height = int(height - 12 - 4)
        rows = deque(maxlen=table_height)
        l1as = {}
        for cycle in range(self.nmin): l1as[cycle] = []

        with Live(layout, vertical_overflow="crop") as live:
            progress.start_task(sim_task)
            self.start_timer = True
            while self.SIM:
                l1a_rate_cnt = self.kcu.read_node("SYSTEM.L1A_RATE_CNT").value()/1000.0
                time_stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cycles = self.cycles
                occupancy = self.kcu.read_node("READOUT_BOARD_0.RX_FIFO_OCCUPANCY")
                temps = self.rb.read_temp()
                lost = self.kcu.read_node(f"READOUT_BOARD_0.RX_FIFO_LOST_WORD_CNT")
                packet_rate = self.kcu.read_node(f"READOUT_BOARD_0.PACKET_RX_RATE").value()/1000.0

                rows.append((f"[bold red]{time_stamp}", f"{cycles}", f"{l1a_rate_cnt:.2f}", f"{occupancy}", f"{temps['t1']:.2f}", f"{temps['t2']:.2f}", f"{temps['t_SCA']:.2f}", f"{temps['t_VTRX']:.2f}", f"{lost}", f"{packet_rate:.2f}"))
                if l1a_rate_cnt != 0: l1as[cycles].append(l1a_rate_cnt)

                layout["dynamic"].update(generate_table(rows))
                layout["progress"].update(Panel(Align.center(progress), expand=True))
                layout["header"].update(generate_header())
                layout["static"].update(generate_static(l1as, self.l1a_rate, self.nmin))
                layout["files"].update(generate_files(self))

                progress.update(sim_task, advance=1)
                time.sleep(1)

