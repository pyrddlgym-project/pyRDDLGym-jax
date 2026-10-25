# ***********************************************************************
# JAXPLAN
#
# Author: Michael Gimelfarb
#
# ***********************************************************************


import importlib.util
import multiprocessing
import threading
import time
from typing import Optional

from rich import get_console
from rich.live import Live
from rich.segment import Segment
from rich.style import Style

STATUS_STYLES = {'success': Style(color='green'), 'info': Style(color='cyan'),
                 'warning': Style(color='yellow'), 'error': Style(color='red')}
BOLD = Style(bold=True)
DIM = Style(dim=True)
NEWLINE = Segment.line()


def _format_seconds(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours}:{minutes:02d}:{seconds:02d}'


class JaxPlannerProgress:
    '''Reports the progress of the planner during training: a live display that refreshes
    periodically in a terminal or notebook, otherwise a plain line printed periodically
    (e.g. when output is redirected to a file, or in a worker process).

    The training loop only stores references to its latest statistics: all indexing and
    formatting happens in the refresh thread, which draws a frame only when they changed.'''

    def __init__(self, position: Optional[int]=None,
                 check_preconditions: bool=False,
                 refresh_per_second: float=5.,
                 plain_interval: float=5.) -> None:
        '''Creates a new progress reporter.

        :param position: index of this run shown as a label (e.g. for multiprocessing)
        :param check_preconditions: whether to mark returns of feasible plans with F
        :param refresh_per_second: maximum refresh rate of the live display
        :param plain_interval: seconds between plain lines when live display is unsupported
        '''
        self.label = '' if position is None else f'[{position}] '
        self.check_preconditions = check_preconditions
        self.refresh_interval = 1. / refresh_per_second
        self.plain_interval = plain_interval
        self._state = None
        self._last_plain = -float('inf')
        self._started = False
        self._thread = None

        # live display cannot be shared across processes or rendered in a file
        console = get_console()
        if multiprocessing.parent_process() is not None or console.is_dumb_terminal:
            supported = False
        elif console.is_jupyter:
            supported = importlib.util.find_spec('ipywidgets') is not None
        else:
            supported = console.is_terminal
        if supported:
            self._live = Live(self, console=console, auto_refresh=False)
            self._bar_styles = (console.get_style('bar.complete'),
                                console.get_style('bar.finished'),
                                console.get_style('bar.back'))
        else:
            self._live = None

    def start(self) -> None:
        if not self._started:
            self._started = True
            if self._live is not None:
                self._live.start(refresh=True)
                self._done = threading.Event()
                self._thread = threading.Thread(target=self._refresh_loop, daemon=True)
                self._thread.start()

    def stop(self) -> None:
        if self._started:
            self._started = False
            if self._live is not None:
                self._done.set()
                self._thread.join()
                self._live.stop()
            elif self._state is not None:
                print(self._plain_line(self._state), flush=True)

    def update(self, iteration, progress, elapsed, status, train_loss, test_loss,
               test_violations, test_index, best_loss, best_violations, best_found_by,
               aux_loss, aux_violations, aux_index) -> None:
        '''Records references to the latest statistics: arrays must not be modified in
        place afterwards. The aux arguments are None without an auxiliary planner.'''
        self._state = (iteration, progress, elapsed, status, train_loss, test_loss,
                       test_violations, test_index, best_loss, best_violations,
                       best_found_by, aux_loss, aux_violations, aux_index)
        if self._live is None:
            now = time.monotonic()
            if now - self._last_plain >= self.plain_interval:
                self._last_plain = now
                print(self._plain_line(self._state), flush=True)

    def _refresh_loop(self) -> None:
        shown = self._state
        while not self._done.wait(self.refresh_interval):
            state = self._state
            if state is not shown:
                shown = state
                self._live.refresh()

    def _marker(self, violations) -> str:
        '''A return is marked F if its plan satisfies the action preconditions.'''
        if not self.check_preconditions:
            return ''
        return 'F' if violations == 0 else ' '

    def _columns(self, state):
        (_, _, _, _, train_loss, test_loss, test_violations, test_index, best_loss,
         best_violations, best_found_by, aux_loss, aux_violations, aux_index) = state
        names = ['train', 'test']
        values = [f'{-train_loss[test_index]:.5f}',
                  f'{-test_loss[test_index]:.5f}{self._marker(test_violations[test_index])}']
        if aux_loss is not None:
            names.append('aux')
            values.append(f'{-aux_loss[aux_index]:.5f}'
                          f'{self._marker(aux_violations[aux_index])}')
        names.append('best')
        values.append(f'{best_found_by} {-best_loss:.5f}{self._marker(best_violations)}')
        return names, values

    def _plain_line(self, state) -> str:
        it, progress, elapsed, status = state[:4]
        names, values = self._columns(state)
        stats = ' | '.join(f'{value} {name}' for (name, value) in zip(names, values))
        return (f'{self.label}{progress:5.1f}% {_format_seconds(elapsed)} | {it} it | '
                f'{stats} | {status.name} status | {(it + 1) / (elapsed + 1e-6):.2f} it/s')

    def __rich_console__(self, console, options):
        state = self._state
        if state is None:
            yield Segment(f'{self.label}compiling...', DIM)
            yield NEWLINE
            return
        it, progress, elapsed, status = state[:4]

        # progress bar in half cells, filling the width not used by the counters
        counters = f' {progress:3.0f}% {_format_seconds(elapsed)} {it} it '
        speed = f'{(it + 1) / (elapsed + 1e-6):.2f} it/s'
        width = max(10, options.max_width - len(self.label) - len(counters) - len(speed))
        complete, finished, back = self._bar_styles
        if options.ascii_only or options.legacy_windows:
            cell, full, half = '-', int(width * min(progress, 100.) / 100.), 0
        else:
            cell = '━'
            full, half = divmod(int(2 * width * min(progress, 100.) / 100.), 2)
        if self.label:
            yield Segment(self.label, BOLD)
        yield Segment(cell * full, finished if progress >= 100. else complete)
        if half:
            yield Segment('╸', complete)
        yield Segment(cell * (width - full - half), back)
        yield Segment(counters)
        yield Segment(speed, DIM)
        yield NEWLINE

        # table of returns and status in columns aligned to the widest entry
        names, values = self._columns(state)
        widths = [max(len(name), len(value)) for (name, value) in zip(names, values)]
        indent = ' ' * len(self.label)
        yield Segment(indent + '  '.join(
            name.rjust(w) for (name, w) in zip(names, widths)) + '  status', BOLD)
        yield NEWLINE
        yield Segment(indent + '  '.join(
            value.rjust(w) for (value, w) in zip(values, widths)) + '  ')
        yield Segment(status.name, STATUS_STYLES.get(status.get_type()))
        yield NEWLINE
