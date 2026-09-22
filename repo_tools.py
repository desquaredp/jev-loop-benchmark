"""Small stdlib tool runner, executed INSIDE a disposable benchmark container."""
import fnmatch
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile

OUTPUT_LIMIT = 16000


def clip(text, limit=OUTPUT_LIMIT):
    if len(text) <= limit:
        return text
    return text[:limit // 2] + '\n[output truncated; narrow the query or page the file]\n' + text[-limit // 2:]


def git(*args):
    return subprocess.check_output(['git', *args], universal_newlines=True, encoding='utf-8', timeout=15)


def files():
    return sorted(set(git('ls-files', '-z', '--cached', '--others', '--exclude-standard').split('\0')) - {''})


def path_in_repo(name):
    if not isinstance(name, str) or not name or '\n' in name or '\0' in name:
        raise ValueError('Expected a repository-relative filename')
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or '.git' in path.parts:
        raise ValueError('Paths must stay inside the repository; no Git internals')
    root = Path.cwd().resolve()
    if root not in path.resolve().parents:
        raise ValueError('Path escapes repository (including through a symlink)')
    return path


def is_test(path):
    return any(p in ('tests', 'testing', 'test') for p in Path(path).parts) or Path(path).name.startswith('test_')


def command(argv, timeout):
    """Bound test runtime AND output; kill the process group, not just its parent."""
    if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
        raise ValueError('test.argv must be a nonempty list of strings, not a shell command')
    timeout = max(1, min(int(timeout), 60))
    with tempfile.TemporaryFile() as log:
        proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)
        # A test may spawn children and exit first. Do not leave them running.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        size = log.tell()
        log.seek(0)
        head = log.read(OUTPUT_LIMIT // 2).decode(errors='replace')
        log.seek(max(OUTPUT_LIMIT // 2, size - OUTPUT_LIMIT // 2))
        tail = log.read(OUTPUT_LIMIT // 2).decode(errors='replace')
    return {'status': 'test_timeout' if timed_out else 'ok',
            'exit_code': 124 if timed_out else proc.returncode,
            'output': head + ('\n[output truncated]\n' if size > OUTPUT_LIMIT else '') + tail,
            'timeout_seconds': timeout, 'output_bytes': size}


def dispatch(action, args):
    if not isinstance(args, dict):
        raise ValueError('args must be an object')
    if action == 'files':
        matches = [p for p in files() if fnmatch.fnmatch(p, args.get('glob', '*'))]
        offset = max(0, int(args.get('offset', 0)))
        return {'paths': matches[offset:offset + 100], 'total': len(matches),
                'next_offset': offset + 100 if len(matches) > offset + 100 else None}
    if action == 'read':
        path = path_in_repo(args['path'])
        lines = path.read_text(encoding='utf-8',errors='replace').splitlines(keepends=True)
        start = max(1, int(args.get('start_line', 1)))
        end = min(len(lines), start + 399, int(args.get('end_line', start + 199)))
        if end < start and lines:
            raise ValueError('Requested range is empty; check total_lines or start_line')
        text = ''.join(lines[start - 1:end])
        return {'path': str(path), 'start_line': start, 'end_line': end,
                'total_lines': len(lines), 'content': clip(text),
                'truncated': len(text) > OUTPUT_LIMIT,
                'next_line': end + 1 if end < len(lines) else None}
    if action == 'search':
        query = args['query']
        if not isinstance(query, str) or not query:
            raise ValueError('search.query must be nonempty literal text')
        pattern = args.get('glob', '*')
        offset = max(0, int(args.get('offset', 0)))
        hits = []
        for name in files():
            if not fnmatch.fnmatch(name, pattern):
                continue
            path = path_in_repo(name)
            if not path.is_file() or path.stat().st_size > 2_000_000:
                continue
            text = path.read_text(encoding='utf-8',errors='replace')
            if '\0' in text:
                continue
            for number, line in enumerate(text.splitlines(), 1):
                if query.lower() in line.lower():
                    hits.append({'path': name, 'line': number, 'text': line[:240]})
        return {'matches': hits[offset:offset + 40], 'total': len(hits),
                'next_offset': offset + 40 if len(hits) > offset + 40 else None}
    if action == 'edit':
        edits = args['edits']
        if not isinstance(edits, list) or not edits:
            raise ValueError('edit.edits must be a nonempty list')
        pending = {}
        new_paths = []
        for edit in edits:
            path = path_in_repo(edit['path'])
            if is_test(path):
                raise ValueError('Do not modify benchmark tests; use test with python -c for an extra repro')
            if path not in pending:
                if path.exists():
                    pending[path] = path.read_text(encoding='utf-8')
                else:
                    pending[path] = ''
                    new_paths.append(path)
            old, new = edit['old'], edit['new']
            if not isinstance(old, str) or not isinstance(new, str):
                raise ValueError('old/new must be strings')
            if not old:
                if path not in new_paths or pending[path]:
                    raise ValueError('Empty old is allowed only for creating a new file')
                pending[path] = new
            else:
                if pending[path].count(old) != 1:
                    raise ValueError('old text must match exactly once in CURRENT source: ' + str(path))
                pending[path] = pending[path].replace(old, new, 1)
        # Validate the complete batch before making any edits.
        for path, text in pending.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text,encoding='utf-8')
        for path in new_paths:
            git('add', '-N', '--', str(path))
        return {'edited': [str(p) for p in pending], 'exit_code': 0}
    if action == 'test':
        return command(args['argv'], args.get('timeout_seconds', 60))
    if action == 'catalog':
        # Fresh, cheap lexical retrieval for JEV, NOT the old cached top-20 fence.
        words = {w.lower() for w in re.findall(r'[A-Za-z_][A-Za-z_0-9]{3,}', args['issue'])}
        words -= {'this', 'that', 'with', 'from', 'when', 'have', 'should', 'would', 'return', 'self', 'none', 'true', 'false', 'using', 'then', 'test', 'tests'}
        exact = set(args.get('requested_paths', []))
        ranked = []
        for name in files():
            if not name.endswith('.py') or is_test(name):
                continue
            path = path_in_repo(name)
            if not path.is_file() or path.stat().st_size > 500_000:
                continue
            text = path.read_text(encoding='utf-8',errors='replace')
            present = words & set(re.findall(r'[a-z_][a-z_0-9]{3,}', text.lower()))
            score = len(present) + 4 * sum(w in name.lower() for w in words) + (1000 if name in exact else 0)
            if not score:
                continue
            lines = text.splitlines()
            center = max(range(len(lines)), key=lambda i: sum(w in lines[i].lower() for w in present)) if lines else 0
            ranked.append((score, name, {'path': name, 'line': center + 1,
                'excerpt': '\n'.join(lines[max(0, center - 4):center + 12])[:1200]}))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return {'candidates': [r[2] for r in ranked[:40]], 'total_scored': len(ranked)}
    raise ValueError('Unknown tool: ' + str(action))


if __name__ == '__main__':
    try:
        request = json.load(sys.stdin)
        result = dispatch(request['action'], request.get('args', {}))
        print(json.dumps({'status': 'ok', **result}))
    except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError) as exc:
        print(json.dumps({'status': 'tool_error', 'error': str(exc)}))
