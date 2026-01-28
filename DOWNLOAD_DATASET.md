# How to Download Full ARC-AGI Dataset

## Option 1: Clone Official Repository (RECOMMENDED)

```bash
# Clone the official ARC-AGI repository
git clone https://github.com/fchollet/ARC-AGI.git

# Copy the data folder to your project
cp -r ARC-AGI/data/* d:/arc_agi/data/
```

After this you should have:
- `data/training/` - 400 training tasks
- `data/evaluation/` - 400 evaluation tasks

## Option 2: Download ARC Prize 2024/2025 Data

```bash
# Clone the ARC Prize repository
git clone https://github.com/arcprize/arc-prize-2025.git

# Copy the data
cp -r arc-prize-2025/data/* d:/arc_agi/data/
```

## Option 3: Direct Download (Manual)

1. Go to: https://github.com/fchollet/ARC-AGI
2. Click "Code" -> "Download ZIP"
3. Extract and copy `data/` folder to `d:/arc_agi/data/`

## File Structure Expected

```
d:/arc_agi/data/
├── training/
│   ├── 007bbfb7.json
│   ├── 00d62c1b.json
│   └── ... (400 files)
└── evaluation/
    ├── 0a938d79.json
    └── ... (400 files)
```

## Verify Installation

```bash
python -c "import os; print(f'Training tasks: {len(os.listdir(\"data/training\"))}')"
python -c "import os; print(f'Evaluation tasks: {len(os.listdir(\"data/evaluation\"))}')"
```

## Run Benchmark

```bash
# Test on training set
python test_arc_comprehensive.py

# Or test on specific subset
python -c "from src.arc.solver import run_arc_file_benchmark; run_arc_file_benchmark('data/training')"
```
