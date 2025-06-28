from mt_dataset.combination import Fashion200kCombination

path = "C:\\Users\\user\\fashion200k"
dataset = Fashion200kCombination(path)

print(len(dataset))
n_turn3 = [transaction for transaction in dataset if transaction["n_turns"] == 3]

n_turn4 = [transaction for transaction in dataset if transaction["n_turns"] == 4]

n_turn5 = [transaction for transaction in dataset if transaction["n_turns"] == 5]

