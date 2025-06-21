from mt_dataset.convergence import Fashion200kConvergence

path = "C:\\Users\\user\\fashion200k"
dataset = Fashion200kConvergence(path)

print(len(dataset))

n_turn3 = [transaction for transaction in dataset if transaction["n_turns"] == 3]
print(n_turn3)

n_turn4 = [transaction for transaction in dataset if transaction["n_turns"] == 4]
print(n_turn4)

n_turn5 = [transaction for transaction in dataset if transaction["n_turns"] == 5]
print(n_turn5)
