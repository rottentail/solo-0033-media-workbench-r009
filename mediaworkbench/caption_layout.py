"""Exact bottleneck reading-rate optimization of bounded two-line captions.

The whole programme is optimized at once.  A cue never crosses an occurrence
(clip) boundary, but the cue-count budget and the worst reading rate are
decided globally rather than per clip, so a dense passage can spend the cue
budget that would otherwise be wasted on loose early sentences.  All reading
rates stay exact rationals.
"""
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


def _min_cue_count(edges, token_count, max_cues, limit):
    """Fewest cues reaching token_count on edges no faster than limit, capped."""
    best = {0: 0}
    for position in range(token_count):
        count = best.get(position)
        if count is None or count >= max_cues:
            continue
        for stop, rate, _rag, _cue in edges[position]:
            if rate > limit:
                continue
            candidate = count + 1
            previous = best.get(stop)
            if previous is None or candidate < previous:
                best[stop] = candidate
    return best.get(token_count)


def _best_path(edges, token_count, max_cues, limit):
    """Exact programme-wide optimum among paths whose edges respect limit."""
    # State key carries the full contract tie-break order:
    # total squared unused cells, cue count, cue-ending word index tuple.
    state = {0: ((0, 0, ()), None)}
    for position in range(token_count):
        entry = state.get(position)
        if entry is None:
            continue
        key, _back = entry
        if key[1] >= max_cues:
            continue
        for edge in edges[position]:
            stop, rate, rag, cue = edge
            if rate > limit:
                continue
            new_key = (key[0] + rag, key[1] + 1, key[2] + (stop,))
            old = state.get(stop)
            if old is None or new_key < old[0]:
                state[stop] = (new_key, (position, edge))
    if token_count not in state:
        return None
    path = []
    position = token_count
    while position:
        previous, edge = state[position][1]
        path.append(edge)
        position = previous
    path.reverse()
    return path


def solve(tokens, duration=0, width=32, max_cues=32, minimum_ms=400, maximum_ms=5000, hold_ms=500):
    bounds = ((width, 4, 80), (max_cues, 1, 128), (minimum_ms, 1, 10000),
              (maximum_ms, minimum_ms, 10000), (hold_ms, 0, 2000))
    for value, low, high in bounds:
        if type(value) is not int or not low <= value <= high:
            raise ValueError('caption width, count or duration bounds are invalid')
    if not tokens:
        return {'cues': [], 'worst_rate': [0, 1], 'raggedness': 0}
    edges = candidates(tokens, duration, width, minimum_ms, maximum_ms, hold_ms)
    rates = sorted({rate for outgoing in edges.values() for _stop, rate, _r, _c in outgoing})
    if not rates:
        raise ValueError('no complete caption layout fits the device limits')
    lo, hi = 0, len(rates) - 1
    if _min_cue_count(edges, len(tokens), max_cues, rates[hi]) is None:
        raise ValueError('no complete caption layout fits the device limits')
    while lo < hi:
        mid = (lo + hi) // 2
        if _min_cue_count(edges, len(tokens), max_cues, rates[mid]) is not None:
            hi = mid
        else:
            lo = mid + 1
    threshold = rates[lo]
    path = _best_path(edges, len(tokens), max_cues, threshold)
    if not path or path[-1][0] != len(tokens):
        raise ValueError('caption layout did not cover all words')
    return {'cues': [edge[3] for edge in path],
            'worst_rate': [threshold.numerator, threshold.denominator],
            'raggedness': sum(edge[2] for edge in path)}
