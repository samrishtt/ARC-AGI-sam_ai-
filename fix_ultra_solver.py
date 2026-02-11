
path = 'src/arc/ultra_solver.py'

with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
skip = False
for line in lines:
    if line.strip() == 'class PrimitiveDSL:':
        skip = True
    
    if skip and line.strip() == 'class ColorMapper:':
        skip = False
    
    if not skip:
        new_lines.append(line)

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print(f"Fixed {path}. Original lines: {len(lines)}, New lines: {len(new_lines)}")
