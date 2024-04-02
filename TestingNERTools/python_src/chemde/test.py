import os
from pprint import pprint  # pretty printer

import csv

import Levenshtein
from chemdataextractor import Document

from TestingNERTools.python_src.metrics import (
    calculate_precision,
    calculate_recall,
    calculate_f1_score,
    levenshtein_distance,
)


# client.api_key = config.OPENAI_API_KEY


def getChemDEResult(text):
    doc = Document(text)
    entities = doc.cems

    result = [str(item) for item in entities]
    return result


# Specify the path to your CSV file
csv_file_path = "dataset csv file"

# Open the CSV file in read mode
with open(csv_file_path, "r", encoding="cp1252") as file:
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

        text = row["text"]
        entities = set(getChemDEResult(text))
        # You can access specific columns using the column names, e.g., row['ColumnName']
        goldenEntities = set(
            [item.strip().lower() for item in row["chemicals"].split(",")]
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
        print("CHEMDE: ", entities)
        print(
            "Custom: ",
            set([item.strip().lower() for item in row["chemicals"].split("; ")]),
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
