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

def input_int_with_default(prompt, default):
    while True:
        val = input(prompt).strip()
        if val == "":
            return default
        try:
            intval = int(val)
            if intval > 0:
                return intval
            else:
                print("Please enter a positive integer.")
        except ValueError:
            print("Invalid input. Please enter a positive integer or press Enter for default.")

def main():
    print("Available encoding methods:")
    for i, method in enumerate(methods, 1):
        print(f"  {i}. {method}")

    choice = input("\nEnter method numbers to run (e.g. 1,3,5 or 2-4). Press Enter to run ALL: ").strip()
    selected_indices = parse_choices(choice, len(methods))

    if not selected_indices:
        print("No valid selections. Running all methods by default.")
        selected_indices = set(range(1, len(methods) + 1))

    # Prompt batch_size and num_epochs only once if method 5 is selected
    batch_size = None
    num_epochs = None
    if 5 in selected_indices:
        batch_size = input_int_with_default("Enter BATCH_SIZE (default 4): ", 4)
        num_epochs = input_int_with_default("Enter NUM_EPOCHS (default 10): ", 10)

    for i in sorted(selected_indices):
        method_script = os.path.join(methods_folder, methods[i - 1])
        if not os.path.exists(method_script):
            print(f"[WARNING] Script '{method_script}' not found.")
            continue

        print(f"\n--- Running: {method_script} ---")

        try:
            if i == 5:
                subprocess.run(
                    ["python", method_script, str(batch_size), str(num_epochs)],
                    check=True
                )
            else:
                subprocess.run(["python", method_script], check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Script {method_script} failed with error: {e}")

if __name__ == "__main__":
    main()
