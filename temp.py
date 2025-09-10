from pathlib import Path
import json
import random
from mt_dataset.fashion_prompts import replacement, addition

json_path = "/home/gpuadmin/dasol/mt-fashion200k/mt_dataset"

with open(Path(json_path) / "test_convergence.json",
          encoding="utf-8") as convergence:
    convergence = json.load(convergence)

with open(Path(json_path) / "test_rollback2.json", encoding="utf-8") as rollback:
    rollback = json.load(rollback)
with open(Path(json_path) / "test_combination2.json", encoding="utf-8") as combination:
    combination = json.load(combination)

rollback_result = []
for sample in rollback:
    n_turns = sample["n_turns"]
    last_turn = f"turn-{n_turns}"
    target_word = sample[last_turn]["target_word"]
    if sample[last_turn]["source_word"] == "":
        template = random.choice(addition)
        mod_str = f"Turn {n_turns}: " + template.format(new_attr=target_word)
    else:
        source_word = sample[last_turn]["source_word"]
        template = random.choice(replacement)
        mod_str = f"Turn {n_turns}: " + template.format(old_attr=source_word, new_attr=target_word)
    sample[last_turn]["mod_str"] = mod_str
    rollback_result.append(sample)
with open(f"test_rollback.json", "w", encoding="utf-8") as f:
    json.dump(rollback_result, f, indent=4, ensure_ascii=False)


combination_result = []
for sample in combination:
    n_turns = sample["n_turns"]
    last_turn = f"turn-{n_turns}"
    target_word = sample[last_turn]["target_word"]
    if sample[last_turn]["source_word"] == "":
        template = random.choice(addition)
        mod_str = f"Turn {n_turns}: " + template.format(new_attr=target_word)
    else:
        source_word = sample[last_turn]["source_word"]
        template = random.choice(replacement)
        mod_str = f"Turn {n_turns}: " + template.format(old_attr=source_word, new_attr=target_word)
    sample[last_turn]["mod_str"] = mod_str
    combination_result.append(sample)
with open(f"test_combination.json", "w", encoding="utf-8") as f:
    json.dump(combination_result, f, indent=4, ensure_ascii=False)