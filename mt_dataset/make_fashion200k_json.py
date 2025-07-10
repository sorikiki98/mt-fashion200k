from torch.utils.data import ConcatDataset
from convergence import Fashion200kConvergence
from rollback import Fashion200kRollback
from combination import Fashion200kCombination
from conflict import Fashion200kConflict
from fashion200k_utils import export_transactions_json

path = "C:\\Users\\user\\fashion200k"

# convergence
convergence_train = Fashion200kConvergence(path, seed=42, split="train")
export_transactions_json(convergence_train, split="train", name="convergence")
convergence_test = Fashion200kConvergence(path, seed=42, split="test")
export_transactions_json(convergence_test, split="test", name="convergence")

# rollback
rollback_train = Fashion200kRollback(path, seed=71, split="train")
export_transactions_json(rollback_train, split="train", name="rollback")
rollback_test = Fashion200kRollback(path, seed=71, split="test")
export_transactions_json(rollback_test, split="test", name="rollback")

# conflict
conflict_train = Fashion200kConflict(path, seed=210, split="train")
export_transactions_json(conflict_train, split="train", name="conflict")
conflict_test = Fashion200kConflict(path, seed=210, split="test")
export_transactions_json(conflict_test, split="test", name="conflict")

# combination
combination_train1 = Fashion200kCombination(path, seed=155, split="train")
combination_train2 = Fashion200kCombination(path, seed=165, split="train")
combination_train3 = Fashion200kCombination(path, seed=175, split="train")
combination_train_set = ConcatDataset([combination_train1, combination_train2, combination_train3])
export_transactions_json(combination_train_set, split="train", name="combination")

combination_test1 = Fashion200kCombination(path, seed=155, split="test")
combination_test2 = Fashion200kCombination(path, seed=165, split="test")
combination_test3 = Fashion200kCombination(path, seed=175, split="test")
combination_test_set = ConcatDataset([combination_test1, combination_test2, combination_test3])
export_transactions_json(combination_test_set, split="test", name="combination")
