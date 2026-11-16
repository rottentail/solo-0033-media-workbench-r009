"""Exact bottleneck reading-rate optimization of bounded two-line captions."""
from fractions import Fraction
import unicodedata


def cells(text):
    return sum(0 if unicodedata.category(ch).startswith('M') else
               2 if unicodedata.east_asian_width(ch) in ('W', 'F') else 1 for ch in text)


def lines(words, width):
    text = [word['text'] for word in words]
    choices = []
    for split in range(1, len(text)+1):
        rows = [' '.join(text[:split])]
        if split < len(text):
            rows.append(' '.join(text[split:]))
        lengths = [cells(row) for row in rows]
        if all(length <= width for length in lengths):
            choices.append((sum((width-length)**2 for length in lengths), split, rows))
    return min(choices) if choices else None


def candidates(tokens, duration, width, minimum_ms, maximum_ms, hold_ms):
    edges = {index: [] for index in range(len(tokens))}
    for start, first in enumerate(tokens):
        for stop in range(start+1, min(len(tokens), start+32)+1):
            last = tokens[stop-1]
            if last['occurrence'] != first['occurrence']:
                break
            layout = lines(tokens[start:stop], width)
            if layout is None:
                continue
            next_start = tokens[stop]['start_ms'] if stop < len(tokens) else duration
            end = min(last['end_ms']+hold_ms, first['start_ms']+maximum_ms,
                      first['clip_end_ms'], next_start)
            if end < last['end_ms'] or end-first['start_ms'] < minimum_ms:
                continue
            cue = {'start_ms': first['start_ms'], 'end_ms': end, 'lines': layout[2],
                   'word_start': start, 'word_stop': stop, 'occurrence': first['occurrence']}
            units = sum(cells(word['text']) for word in tokens[start:stop])
            edges[start].append((stop, Fraction(units*1000, end-first['start_ms']), layout[0], cue))
    return edges


def greedy_path(edges, start, stop, quota):
    path = []
    position = start
    while position < stop and len(path) < quota:
        choices = [edge for edge in edges[position] if edge[0] <= stop]
        remaining = quota-len(path)
        if remaining == 1:
            choices = [edge for edge in choices if edge[0] == stop]
        if not choices:
            return None
        edge = min(choices, key=lambda edge: (edge[1], edge[2], -edge[0]))
        path.append(edge)
        position = edge[0]
    return path if position == stop else None


def solve(tokens, duration, width=32, max_cues=32, minimum_ms=400, maximum_ms=5000, hold_ms=500):
    bounds = ((width, 4, 80), (max_cues, 1, 128), (minimum_ms, 1, 10000),
              (maximum_ms, minimum_ms, 10000), (hold_ms, 0, 2000))
    for value, low, high in bounds:
        if type(value) is not int or not low <= value <= high:
            raise ValueError('caption width, count or duration bounds are invalid')
    if not tokens:
        return {'cues': [], 'worst_rate': [0, 1], 'raggedness': 0}
    edges = candidates(tokens, duration, width, minimum_ms, maximum_ms, hold_ms)
    groups = []
    start = 0
    while start < len(tokens):
        stop = start+1
        while stop < len(tokens) and tokens[stop]['occurrence'] == tokens[start]['occurrence']:
            stop += 1
        groups.append((start, stop))
        start = stop
    selected = []
    available = max_cues
    for group_index, (start, stop) in enumerate(groups):
        groups_left = len(groups)-group_index-1
        share = max(1, round(max_cues*(stop-start)/len(tokens)))
        quota = min(share, available-groups_left)
        if quota < 1:
            raise ValueError('no complete caption layout fits the device limits')
        choices = []
        for count in range(1, quota+1):
            path = greedy_path(edges, start, stop, count)
            if path is None:
                continue
            score = (max(edge[1] for edge in path), sum(edge[2] for edge in path),
                     len(path), tuple(edge[0] for edge in path))
            choices.append((score, path))
        if not choices:
            raise ValueError('no complete caption layout fits the device limits')
        score, path = min(choices, key=lambda candidate: candidate[0])
        selected.extend(path)
        available -= len(path)
    if not selected or selected[-1][0] != len(tokens):
        raise ValueError('caption layout did not cover all words')
    threshold = max(edge[1] for edge in selected)
    result = {'cues': [edge[3] for edge in selected],
              'worst_rate': [threshold.numerator, threshold.denominator],
              'raggedness': sum(edge[2] for edge in selected)}
    return result
