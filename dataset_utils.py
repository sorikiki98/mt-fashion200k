import csv


def extract_siblings_per_category(words_dict, category):
    filtered_words_dict = {parent: value[category] for parent, value in words_dict.items() if category in value}
    return filtered_words_dict


def export_siblings_per_category(words_dict, category):
    csv_file_name = f"parent2different_words_{category}.csv"

    with open(csv_file_name, mode='w', newline='', encoding='utf-8-sig') as f:

        writer = csv.writer(f)
        max_values_len = max(len(v) for v in words_dict.values())

        header = ['key'] + [f'value{i + 1}' for i in range(max_values_len)]
        writer.writerow(header)

        for key, values in words_dict.items():
            if len(values) >= 1:
                row = [key] + values
                row += [''] * (max_values_len - len(values))
                writer.writerow(row)