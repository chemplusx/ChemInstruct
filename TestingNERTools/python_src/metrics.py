import Levenshtein


def calculate_precision(true_positives, false_positives):
    if true_positives + false_positives == 0:
        return 0  # To handle the case when precision is undefined (division by zero)

    precision = true_positives / (true_positives + false_positives)
    return precision


def calculate_recall(true_positives, false_negatives):
    # true_positives = sum(a == 1 and p == 1 for a, p in zip(actual_labels, predicted_labels))
    # false_negatives = sum(a == 1 and p == 0 for a, p in zip(actual_labels, predicted_labels))

    if true_positives + false_negatives == 0:
        return 0  # To handle the case when recall is undefined (division by zero)

    recall = true_positives / (true_positives + false_negatives)
    return recall


def calculate_f1_score(true_positives, false_positives, false_negatives):
    precision = calculate_precision(true_positives, false_positives)
    recall = calculate_recall(true_positives, false_negatives)

    if precision + recall == 0:
        return 0  # To handle the case when F1 score is undefined (division by zero)

    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


# Function to calculate Levenshtein distance
def levenshtein_distance(str1, str2):
    return Levenshtein.distance(str1, str2)
