from mt_dataset.combination import Fashion200kCombination
from torch.utils.data import ConcatDataset

path = "C:\\Users\\user\\fashion200k"
dataset1 = Fashion200kCombination(path, seed=155)
dataset2 = Fashion200kCombination(path, seed=165)
dataset3 = Fashion200kCombination(path, seed=175)
combined_dataset = ConcatDataset([dataset1, dataset2, dataset3])

print(len(combined_dataset))
n_turn3 = [transaction for transaction in combined_dataset if transaction["n_turns"] == 3]

n_turn4 = [transaction for transaction in combined_dataset if transaction["n_turns"] == 4]

n_turn5 = [transaction for transaction in combined_dataset if transaction["n_turns"] == 5]

