"""Replace all non-ASCII chars in run_benchmark.py with ASCII equivalents."""
replacements = {
    '\u2500': '-',   # ─  box-drawing horizontal
    '\u2501': '-',   # ━
    '\u2502': '|',   # │
    '\u2014': '--',  # —  em dash
    '\u2013': '-',   # –  en dash
    '\u2192': '->',  # →  right arrow
    '\u2190': '<-',  # ←
    '\u2714': '[v]', # ✔
    '\u2718': '[x]', # ✘
    '\u2713': '[v]', # ✓
    '\u2717': '[x]', # ✗
    '\u00e9': 'e',   # é
}

with open('run_benchmark.py', encoding='utf-8') as f:
    content = f.read()

fixed = content
for bad, good in replacements.items():
    fixed = fixed.replace(bad, good)

# Verify no remaining non-ASCII
bad_chars = [(i, hex(ord(c)), c) for i, c in enumerate(fixed) if ord(c) > 127]
if bad_chars:
    print(f'Still {len(bad_chars)} non-ASCII chars:')
    for pos, code, ch in bad_chars[:10]:
        print(f'  pos {pos}: {code} = {repr(ch)}')
else:
    with open('run_benchmark.py', 'w', encoding='utf-8') as f:
        f.write(fixed)
    print(f'[OK] run_benchmark.py fully ASCII-safe. ({len(content)} -> {len(fixed)} chars)')
