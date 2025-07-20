from torch.utils.data import ConcatDataset
from convergence import Fashion200kConvergence
from combination import Fashion200kCombination
from rollback import Fashion200kRollback
from fashion200k_utils import export_transactions_json

path = "/mnt/c/Users/user/fashion200k"

# convergence
convergence_train = Fashion200kConvergence(path, seed=42, split="train")
export_transactions_json(convergence_train, split="train", name="convergence")
convergence_test = Fashion200kConvergence(path, seed=42, split="test")
export_transactions_json(convergence_test, split="test", name="convergence")

# combination
combination_train = Fashion200kCombination(path, seed=155, split="train")
export_transactions_json(combination_train, split="train", name="combination")
combination_test = Fashion200kCombination(path, seed=155, split="test")
export_transactions_json(combination_test, split="test", name="combination")

# rollback
rollback_train = Fashion200kRollback(path, seed=71, split="train")
export_transactions_json(rollback_train, split="train", name="rollback")
rollback_test = Fashion200kRollback(path, seed=71, split="test")
export_transactions_json(rollback_test, split="test", name="rollback")


