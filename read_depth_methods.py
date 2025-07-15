import os
import subprocess

methods_folder = "methods"

methods = [
    "depth_hsv_rgb.py",
    "depth_luv_rgb.py",
    "depth_lab_rgb.py",
    "depth_mvd_roi.py",
    "depth_n_depth.py",
    "depth_hybrid_lab_hsv.py"
]

print("Available encoding methods:")
for i, method in enumerate(methods, 1):
    print(f"  {i}. {method}")

choice = input("\nEnter method numbers to run (e.g. 1,3,5 or 2-4). Press Enter to run ALL: ").strip()

def parse_choices(s, max_num):
    selected = set()
    if not s:
        return set(range(1, max_num + 1))  # all selected
    parts = s.split(',')
    for part in parts:
        part = part.strip()
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                if 1 <= start <= end <= max_num:
                    selected.update(range(start, end + 1))
            except ValueError:
                pass
        else:
            try:
                num = int(part)
                if 1 <= num <= max_num:
                    selected.add(num)
            except ValueError:
                pass
    return selected

selected_indices = parse_choices(choice, len(methods))

if not selected_indices:
    print("No valid selections. Running all methods by default.")
    selected_indices = set(range(1, len(methods) + 1))

for i in sorted(selected_indices):
    method_script = os.path.join(methods_folder, methods[i - 1])
    if not os.path.exists(method_script):
        print(f"[WARNING] Script '{method_script}' not found.")
        continue

    print(f"\n--- Running: {method_script} ---")

    # For method 5 (depth_n_depth.py), ask for BATCH_SIZE and NUM_EPOCHS and pass as args
    if i == 5:
        try:
            batch_size = input("Enter BATCH_SIZE (default 4): ").strip()
            if not batch_size:
                batch_size = "4"
            num_epochs = input("Enter NUM_EPOCHS (default 10): ").strip()
            if not num_epochs:
                num_epochs = "10"
        except Exception:
            batch_size = "4"
            num_epochs = "10"

        try:
            subprocess.run(
                ["python", method_script, batch_size, num_epochs],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Script {method_script} failed with error: {e}")

    else:
        # For other methods just run without args (depth file handled internally)
        try:
            subprocess.run(["python", method_script], check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Script {method_script} failed with error: {e}")
