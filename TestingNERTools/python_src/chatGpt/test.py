import os
import config
from pprint import pprint  # pretty printer

import csv

from openai import OpenAI
import Levenshtein


# Function to calculate Levenshtein distance
def levenshtein_distance(str1, str2):
    return Levenshtein.distance(str1, str2)


client = OpenAI(
    api_key=config.OPENAI_API_KEY,
    organization="<your org id>",
)

# client.api_key = config.OPENAI_API_KEY


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


def getChatGptResult(text, gpt4 = False):
    if gpt4:
        model_name = "gpt-4-1106-preview"
    else:
        model_name = "gpt-3.5-turbo",
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user", 
                "content": "You are chemical named entity recognition tool which performs the task of NER. You can identify all the chemical enties given in a text as it is and hold no ability to generate any other information."
            },
            {
                "role": "system",
                "content": "Okay, I will try my best to identify all the chemical entities in the text and will not generate any other information on my own."
            },
            {
                "role": "user",
                "content": "Identify all the chemical entities from the text and give response as comma separated entities. Text: "+ text,
            }
        ],
    )

    result = completion.choices[0].message
    if result.content.find(";") != -1:
        return [item.strip().lower() for item in result.content.split(";")]
    if result.content.find("\n") != -1:
        return [item.strip().lower() for item in result.content.split("\n")]
    else:
        return [item.strip().lower() for item in result.content.split(",")]


# Specify the path to your CSV file
csv_file_path = "<input csv file path here>"

# Open the CSV file in read mode
with open(csv_file_path, "r", encoding="utf8") as file:
    # Create a CSV reader object with a dictionary interface
    csv_reader = csv.DictReader(file)
    gfalsePositives = 0
    gfalseNegatives = 0
    gtruePositives = 0
    gtrueNegatives = 0
    gperfectMatches = 0
    gpartialMatches = 0
    # Iterate over each row in the CSV file
    for row in csv_reader:
        print("--------------------->")
        falsePositives = 0
        falseNegatives = 0
        truePositives = 0
        trueNegatives = 0
        perfectMatches = 0
        partialMatches = 0

        # Each row is a dictionary with column names as keys
        # print(row)

        text = row["Text"]
        # if len(text) > 10000:
        #     part1 = text[: int(len(text)/2) + text[int(len(text)/2):].find(".")]
        #     entities = getChatGptResult(part1)

        #     part2 = text[int(len(text)/2) + text[int(len(text)/2):].find("."):]
        #     entities.extend(getChatGptResult(part2))
        # else:
        entities = getChatGptResult(text)
        # You can access specific columns using the column names, e.g., row['ColumnName']
        goldenEntities = set(
            [item.strip().lower() for item in row["Chemical Entities"].split("; ")]
        )
        for entity in entities:
            minD = 1000000000
            te = ""
            i = 0
            for ge in goldenEntities:
                try:
                    distance = levenshtein_distance(entity.lower(), ge.lower())
                except:
                    te = ge
                    minD = -1
                    break

                # print(f"Levenshtein distance between '{entity}' and '{ge}': {distance}")
                if minD > distance:
                    minD = distance
                    te = ge
                i += 1

            if minD == -1:
                goldenEntities.remove(te)
            if minD <= 2:
                perfectMatches += 1
                truePositives += 1
                goldenEntities.remove(te)
            elif minD <= 4:
                partialMatches += 1
                truePositives += 1
                goldenEntities.remove(te)
            else:
                falsePositives += 1

        falseNegatives += len(goldenEntities)
        print("ChatGpt: ", entities)
        print(
            "NLMChem: ",
            [item.strip().lower() for item in row["Chemical Entities"].split("; ")],
        )
        print("Precision ", calculate_precision(truePositives, falsePositives))
        print("Recall ", calculate_recall(truePositives, falseNegatives))
        print(
            "F1Score ",
            calculate_f1_score(truePositives, falsePositives, falseNegatives),
        )

        gfalsePositives += falsePositives
        gfalseNegatives += falseNegatives
        gtruePositives += truePositives
        gtrueNegatives += trueNegatives
        gperfectMatches += perfectMatches
        gpartialMatches += partialMatches

        print("Global Precision ", calculate_precision(gtruePositives, gfalsePositives))
        print("Global Recall ", calculate_recall(gtruePositives, gfalseNegatives))
        print(
            "Global F1Score ",
            calculate_f1_score(gtruePositives, gfalsePositives, gfalseNegatives),
        )

    print("Final Precision ", calculate_precision(gtruePositives, gfalsePositives))
    print("Final Recall ", calculate_recall(gtruePositives, gfalseNegatives))
    print(
        "Final F1Score ",
        calculate_f1_score(gtruePositives, gfalsePositives, gfalseNegatives),
    )
