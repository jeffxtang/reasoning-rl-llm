import json
import re
import os

# --------------------------
# Parameters
# --------------------------
INPUT_FILE = "../reasoning-rl-baselines/gsm8k_train.jsonl"  # your GSM8K training JSON file
OUTPUT_DIR = "gsm8k_split"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------
# Heuristic functions
# --------------------------
def estimate_steps(solution_text):
    """Estimate reasoning steps from solution string."""
    solution_text = solution_text.split('####')[0]
    lines = solution_text.strip().split('\n')
    steps = 0
    for line in lines:
        if re.search(r'[\+\-\*\/x]', line):
            steps += 1
    return steps

def label_by_steps(solution_text):
    steps = estimate_steps(solution_text)
    if steps <= 2:
        return "easy"
    elif steps <= 4:
        return "medium"
    else:
        return "hard"

def label_by_length(problem_text):
    length = len(problem_text.split())
    if length <= 50:
        return "easy"
    elif length <= 100:
        return "medium"
    else:
        return "hard"

def assign_difficulty(problem_text, solution_text):
    steps_label = label_by_steps(solution_text)
    length_label = label_by_length(problem_text)
    labels = ["easy", "medium", "hard"]
    # take the harder of the two labels
    difficulty = labels[max(labels.index(steps_label), labels.index(length_label))]
    return difficulty

# --------------------------
# Load GSM8K
# --------------------------
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    gsm8k_data = [json.loads(line) for line in f]

# --------------------------
# Split into difficulties
# --------------------------
split_data = {"easy": [], "medium": [], "hard": []}

for entry in gsm8k_data:
    problem_text = entry.get("question", "")
    solution_text = entry.get("answer", "")  # adjust key if GSM8K JSON uses "solution"
    difficulty = assign_difficulty(problem_text, solution_text)
    split_data[difficulty].append(entry)

# --------------------------
# Save outputs
# --------------------------
for difficulty in ["easy", "medium", "hard"]:
    out_file = os.path.join(OUTPUT_DIR, f"{difficulty}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(split_data[difficulty], f, indent=2)

print("Splitting complete:")
for difficulty in ["easy", "medium", "hard"]:
    print(f"{difficulty}: {len(split_data[difficulty])} problems")

