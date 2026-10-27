# ***********************************************************************
# JAXPLAN
#
# Author: Michael Gimelfarb
#
# ***********************************************************************


import importlib.util
import multiprocessing
import numbers
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rich import get_console
from rich.live import Live
from rich.segment import Segment
from rich.style import Style
from rich.text import Text

STATUS_STYLES = {'success': Style(color='green'), 'info': Style(color='cyan'),
                 'warning': Style(color='yellow'), 'error': Style(color='red')}
BOLD = Style(bold=True)
DIM = Style(dim=True)
NEWLINE = Segment.line()


def _format_seconds(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours}:{minutes:02d}:{seconds:02d}'


def _live_supported(console) -> bool:
    '''A live display cannot be shared across processes or rendered in a file.'''
    if multiprocessing.parent_process() is not None or console.is_dumb_terminal:
        return False
    elif console.is_jupyter:
        return importlib.util.find_spec('ipywidgets') is not None
    else:
        return console.is_terminal


def _bar_styles(console):
    return (console.get_style('bar.complete'), console.get_style('bar.finished'),
            console.get_style('bar.back'))


def _bar_segments(width, fraction, styles, options):
    '''A progress bar of the given width in half cells (whole cells in ascii terminals).'''
    complete, finished, back = styles
    fraction = min(fraction, 1.)
    if options.ascii_only or options.legacy_windows:
        cell, full, half = '-', int(width * fraction), 0
    else:
        cell = '━'
        full, half = divmod(int(2 * width * fraction), 2)
    yield Segment(cell * full, finished if fraction >= 1. else complete)
    if half:
        yield Segment('╸', complete)
    yield Segment(cell * (width - full - half), back)


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

        console = get_console()
        if _live_supported(console):
            self._live = Live(self, console=console, auto_refresh=False)
            self._bar_styles = _bar_styles(console)
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
        else:
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
        if self.label:
            yield Segment(self.label, BOLD)
        yield from _bar_segments(width, progress / 100., self._bar_styles, options)
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


def format_value(value: Any) -> str:
    '''Formats a hyper-parameter or return compactly for display.'''
    if isinstance(value, numbers.Integral):
        return str(value)
    elif isinstance(value, numbers.Real):
        return f'{value:.4g}'
    else:
        return str(value)


TUNING_STATE_STYLES = {'complete': Style(), 'best': Style(color='green', bold=True),
                       'pruned': Style(color='yellow'), 'failed': Style(color='red')}


class JaxTuningProgress:
    '''Reports the progress of hyper-parameter tuning: each finished configuration is
    printed as a row of a table that accumulates, below which a live display shows the
    overall progress and the configurations still running.'''

    def __init__(self, tags: Sequence[str], num_configs: int, num_seeds: int,
                 show_seeds: bool=False, refresh_per_second: float=1.) -> None:
        '''Creates a new tuning progress reporter.

        :param tags: names of the tuned hyper-parameters
        :param num_configs: how many configurations will be evaluated
        :param num_seeds: how many training runs evaluate each configuration
        :param show_seeds: whether to show how many runs a configuration finished
        (e.g. when it can be pruned)
        :param refresh_per_second: refresh rate of the live display
        '''
        self.console = get_console()
        self.num_configs = num_configs
        self.num_seeds = num_seeds
        self.refresh_per_second = refresh_per_second
        self.running = {}
        self.finished = 0
        self._start_time = None
        self._live = None

        # columns of the table and their widths: the hyper-parameters have a column each
        # if they fit in the console, otherwise they are listed below each row
        self.tags = list(tags)
        self.columns = [('#', 4), ('state', 8), ('return', 11)]
        if num_seeds > 1:
            self.columns.append(('std', 9))
        if show_seeds:
            self.columns.append(('seeds', 5))
        self.columns.append(('time', 7))
        param_columns = [(tag, max(len(tag), 10)) for tag in self.tags]
        width = sum(w + 2 for (_, w) in self.columns + param_columns)
        if num_seeds > 1:
            width += 7 * num_seeds
        self.wide = width <= self.console.width
        if self.wide:
            self.columns.extend(param_columns)

    # ===========================================================================
    # static output
    # ===========================================================================

    def _row(self, cells: Sequence[str], style: Style, last: str='') -> Text:
        '''A row of the table, followed by the variable-width last column.'''
        row = '  '.join(cell.rjust(width) for (cell, (_, width)) in zip(cells, self.columns))
        if self.num_seeds > 1:
            row += '  ' + last
        return Text(row.rstrip(), style)

    @staticmethod
    def _returns(rewards: Sequence[float]) -> str:
        return ' '.join(f'{reward:.6g}' for reward in rewards)

    def _params(self, values: Sequence[Any]) -> Text:
        '''The hyper-parameters as tag-value pairs, wrapped to the console width.'''
        indent = ' ' * 6
        text, width = Text(indent), len(indent)
        for (tag, value) in zip(self.tags, values):
            pair = len(tag) + 1 + len(format_value(value))
            if width > len(indent) and width + 3 + pair > self.console.width:
                text.append('\n' + indent)
                width = len(indent)
            elif width > len(indent):
                text.append('   ')
                width += 3
            text.append(tag + ' ', DIM)
            text.append(format_value(value))
            width += pair
        return text

    def print_header(self, title: str, subtitle: str,
                     settings: List[List[Tuple[str, Any]]],
                     space: List[Tuple[str, str, str, str]]) -> None:
        '''Prints the settings, the search space and the header of the results table.'''
        console = self.console
        console.print(Text.assemble((title, BOLD), '  ', (subtitle, DIM)))
        for line in settings:
            text = Text('  ')
            for (i, (key, value)) in enumerate(line):
                text.append(('   ' if i else '') + f'{key} ', DIM)
                text.append(str(value))
            console.print(text)

        # search space aligned in columns
        names = ('hyper-parameter', 'range', 'scale', 'config range')
        if not any(row[3] for row in space):
            names, space = names[:3], [row[:3] for row in space]
        widths = [max(len(row[i]) for row in [names] + space) for i in range(len(names))]
        console.print()
        console.print(Text('  ' + '  '.join(
            name.ljust(w) for (name, w) in zip(names, widths)).rstrip(), BOLD))
        for row in space:
            console.print('  ' + '  '.join(
                value.ljust(w) for (value, w) in zip(row, widths)).rstrip(), highlight=False)
        console.print()
        console.print(self._row([name for (name, _) in self.columns], BOLD, 'per seed'))

    def config_finished(self, number: int, state: str, rewards: Optional[Sequence[float]],
                        seconds: float, values: Sequence[Any], is_best: bool,
                        error: Optional[str]=None) -> None:
        '''Prints the row of a finished configuration, and an error message if it failed.'''
        self.running.pop(number, None)
        self.finished += 1
        if rewards:
            mean = sum(rewards) / len(rewards)
            std = (sum((r - mean) ** 2 for r in rewards) / len(rewards)) ** 0.5
            ret, std = f'{mean:.3f}', (f'{std:.3f}' if len(rewards) > 1 else '-')
        else:
            ret, std = '-', '-'
        cells = [str(number), state, ret]
        names = [name for (name, _) in self.columns]
        if 'std' in names:
            cells.append(std)
        if 'seeds' in names:
            cells.append(f'{len(rewards or [])}/{self.num_seeds}')
        cells.append(_format_seconds(seconds))
        if self.wide:
            cells.extend(format_value(value) for value in values)
        style = TUNING_STATE_STYLES['best' if is_best else state]
        self.console.print(self._row(cells, style, self._returns(rewards or [])),
                           highlight=False)
        if not self.wide:
            self.console.print(self._params(values), highlight=False, overflow='ignore',
                               crop=False)
        if error is not None:
            self.console.print(Text('      ' + error, Style(color='red', dim=True)),
                               highlight=False)

    def print_summary(self, elapsed: float, counts: Dict[str, int],
                      best: Optional[Tuple[int, float, Dict[str, Any]]],
                      message: Optional[str]=None) -> None:
        '''Prints the outcome of tuning and the best configuration.'''
        console = self.console
        text = Text.assemble('\n', ('tuning finished', BOLD), f' in {_format_seconds(elapsed)}')
        for (state, count) in counts.items():
            text.append(f'   {count} {state}', TUNING_STATE_STYLES.get(state, Style()))
        console.print(text)
        if message is not None:
            console.print(Text('  ' + message, Style(color='yellow')))
        if best is None:
            console.print(Text('  no configuration completed', Style(color='red')))
            return
        number, value, params = best
        console.print(Text.assemble(
            '  ', ('best', TUNING_STATE_STYLES['best']), f' #{number}  return {value:.3f}'))
        width = max(len(tag) for tag in params) if params else 0
        for (tag, value) in params.items():
            console.print(Text.assemble(
                '    ', (tag.ljust(width), DIM), '  ', format_value(value)), highlight=False)

    # ===========================================================================
    # live output
    # ===========================================================================

    def start(self) -> None:
        self._start_time = time.time()
        if _live_supported(self.console):
            self._bar_styles = _bar_styles(self.console)
            self._live = Live(self, console=self.console, transient=True,
                              refresh_per_second=self.refresh_per_second)
            self._live.start()

    def stop(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    def config_started(self, number: int) -> None:
        self.running[number] = (time.time(), [])

    def seed_finished(self, number: int, seed_index: int, reward: float) -> None:
        '''Records a finished training run of a running configuration (any thread).'''
        state = self.running.get(number)
        if state is not None:
            state[1].append(reward)

    def __rich_console__(self, console, options):
        now = time.time()
        elapsed = now - self._start_time

        # bar of finished configurations, then the counters
        counters = f' {self.finished}/{self.num_configs} configs  {_format_seconds(elapsed)}'
        width = max(10, options.max_width - len(counters))
        yield from _bar_segments(width, self.finished / max(1, self.num_configs),
                                 self._bar_styles, options)
        yield Segment(counters)
        yield NEWLINE

        # one line per running configuration
        for (number, (start, rewards)) in sorted(list(self.running.items())):
            rewards = list(rewards)
            yield Segment(f'  #{number}', BOLD)
            yield Segment(f'  {_format_seconds(now - start)}  '
                          f'{len(rewards)}/{self.num_seeds} seeds', DIM)
            if rewards:
                yield Segment('  ' + self._returns(rewards))
            yield NEWLINE
