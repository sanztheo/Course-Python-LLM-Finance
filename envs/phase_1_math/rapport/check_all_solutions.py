import json
import os
import sys
import glob
import subprocess

# Find all solution notebooks
notebooks = sorted(glob.glob('solutions_*.ipynb'))
python_executable = sys.executable

print(f"Found {len(notebooks)} solution notebooks.")
print(f"Using Python executable: {python_executable}")

failed_notebooks = []

for notebook_path in notebooks:
    print(f"\n{'='*50}")
    print(f"Checking {notebook_path}...")
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        print(f"Failed to read {notebook_path}: {e}")
        failed_notebooks.append((notebook_path, str(e)))
        continue

    code_cells = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            cell_source = cell['source']
            if isinstance(cell_source, str):
                source_lines_raw = cell_source.splitlines(keepends=True)
            else:
                source_lines_raw = cell_source

            source_lines = []
            for line in source_lines_raw:
                # Filter magic commands
                if not line.strip().startswith('%') and not line.strip().startswith('!'):
                    source_lines.append(line)
            source = ''.join(source_lines)
            code_cells.append(source)

    full_code = '\n\n'.join(code_cells)
    
    script_path = f"temp_debug_{notebook_path.replace('.ipynb', '.py')}"
    
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(full_code)
    
    # Run the script
    print(f"Running extracted code...")
    try:
        # Run with timeout to avoid infinite loops, e.g. 30 seconds per notebook should be enough for simple checks
        # But some might be long. Let's give it 60s.
        result = subprocess.run([python_executable, script_path], capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            print(f"❌ FAILED")
            print("Error output:")
            print(result.stderr)
            failed_notebooks.append((notebook_path, result.stderr))
        else:
            print(f"✅ SUCCESS")
            
    except subprocess.TimeoutExpired:
        print(f"❌ TIMEOUT (script took too long)")
        failed_notebooks.append((notebook_path, "Timeout"))
    except Exception as e:
        print(f"❌ EXECUTION ERROR: {e}")
        failed_notebooks.append((notebook_path, str(e)))
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)

print(f"\n{'='*50}")
print("SUMMARY")
if not failed_notebooks:
    print("All notebooks passed successfully! 🎉")
else:
    print(f"{len(failed_notebooks)} notebooks failed:")
    for nb, err in failed_notebooks:
        print(f"- {nb}")
        # Print last few lines of error for context
        lines = err.strip().split('\n')
        last_lines = '\n'.join(lines[-5:]) if len(lines) > 5 else err
        print(f"  Error: {last_lines}\n")
    sys.exit(1)
