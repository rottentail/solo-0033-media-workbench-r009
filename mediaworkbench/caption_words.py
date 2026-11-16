"""Versioned source transcripts and their occurrence-local programme projection."""
import json
import unicodedata
from . import db, timeline


def get(conn, asset_id):
    asset = conn.execute('SELECT metadata,fingerprint FROM assets WHERE id=?', (asset_id,)).fetchone()
    if asset is None:
        raise ValueError('asset not found')
    row = conn.execute('SELECT * FROM caption_transcripts WHERE asset_id=?', (asset_id,)).fetchone()
    return {'asset_id': asset_id, 'revision': row['revision'] if row else 0,
            'fingerprint': asset['fingerprint'], 'words': json.loads(row['words']) if row else []}


def replace(conn, asset_id, words, expected_revision):
    if type(expected_revision) is not int or expected_revision < 0:
        raise ValueError('transcript revision must be a nonnegative integer')
    if not isinstance(words, list) or len(words) > 128:
        raise ValueError('a source transcript supports at most 128 word atoms')
    validated = []
    previous = 0
    for word in words:
        start, end, text = word['start_ms'], word['end_ms'], word['text']
        if (type(start) is not int or type(end) is not int or start < previous or end <= start
                or not isinstance(text, str) or not text or len(text) > 80
                or any(ch.isspace() or unicodedata.category(ch)[0] == 'C' for ch in text)):
            raise ValueError('word atoms must be ordered nonoverlapping intervals and printable text without spaces')
        validated.append({'start_ms': start, 'end_ms': end, 'text': unicodedata.normalize('NFC', text)})
        previous = end
    with db.write(conn):
        current = get(conn, asset_id)
        metadata = json.loads(conn.execute('SELECT metadata FROM assets WHERE id=?', (asset_id,)).fetchone()[0])
        if current['revision'] != expected_revision:
            raise ValueError('transcript revision conflict')
        if previous > round(float(metadata['format']['duration'])*1000):
            raise ValueError('transcript extends beyond the original')
        conn.execute('INSERT INTO caption_transcripts(asset_id,revision,words) VALUES (?,?,?) '
                     'ON CONFLICT(asset_id) DO UPDATE SET revision=excluded.revision,words=excluded.words',
                     (asset_id, expected_revision+1, json.dumps(validated, ensure_ascii=False)))
        result = get(conn, asset_id)
    return result


def capture(conn, project_id):
    # A plain read view: no SAVEPOINT/RELEASE pair, because releasing a
    # savepoint leaves the connection inside an implicit transaction that
    # later rejects a short write transaction on the same connection.
    snapshot = timeline.snapshot(conn, project_id)
    transcripts = {clip['asset_id']: get(conn, clip['asset_id']) for clip in snapshot['clips']}
    return snapshot, transcripts


def source_packets(snapshot, transcripts):
    # One packet per occurrence, not per asset: repeated cuts of the same
    # original keep independent trims, words and projected positions.
    packets = []
    for occurrence, clip in enumerate(snapshot['clips']):
        transcript = transcripts[clip['asset_id']]
        packet = {'occurrence': occurrence, 'asset_id': clip['asset_id'],
                  'revision': transcript['revision'], 'words': []}
        for index, word in enumerate(transcript['words']):
            if word['start_ms'] < clip['start_ms'] or word['end_ms'] > clip['end_ms']:
                continue
            packet['words'].append({'text': word['text'], 'source_index': index,
                'start_ms': word['start_ms']-clip['start_ms'],
                'end_ms': word['end_ms']-clip['start_ms']})
        packets.append(packet)
    return packets


def project(snapshot, transcripts):
    packets = source_packets(snapshot, transcripts)
    tokens = []
    offset = 0
    for occurrence, clip in enumerate(snapshot['clips']):
        length = clip['end_ms']-clip['start_ms']
        packet = packets[occurrence]
        projected = []
        for word in packet['words']:
            if word['end_ms'] > length:
                continue
            token = {'text': word['text'], 'start_ms': offset+word['start_ms'],
                     'end_ms': offset+word['end_ms'], 'clip_end_ms': offset+length,
                     'occurrence': occurrence, 'source_index': word['source_index'],
                     'asset_id': packet['asset_id'], 'transcript_revision': packet['revision']}
            projected.append(token)
        projected.sort(key=lambda word: (word['start_ms'], word['source_index']))
        tokens.extend(projected)
        offset += length
    if not snapshot['clips'] or offset > 600000 or len(tokens) > 128:
        raise ValueError('caption programmes need clips, at most ten minutes and 128 projected words')
    return tokens, offset
