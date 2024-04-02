import pathlib

from pprint import pprint  # pretty printer

import csv
import Levenshtein
import textwrap
import config
import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown

from TestingNERTools.python_src.metrics import (
    calculate_precision,
    calculate_recall,
    calculate_f1_score,
    levenshtein_distance,
)

genai.configure(api_key=config.API_KEY)


def to_markdown(text):
    text = text.replace("•", "  *")
    return Markdown(textwrap.indent(text, "> ", predicate=lambda _: True))


def checkAvailableModels():
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            print(m.name)


model = genai.GenerativeModel("gemini-pro")


# client.api_key = config.OPENAI_API_KEY


def getGeminiResult(text):
    chat = model.start_chat(history=[])
    prompt = (
        f"Identify and list only the specific chemical names mentioned in the following text. Text: "
        + text
    )
    response = chat.send_message(prompt)
    print("Output of firsdt prompt: \n", response.text)

    formatted_response = chat.send_message(
        "Format the above response as only the entities, semicolon separated."
    )
    print("Output of second prompt: ", formatted_response.text)
    #     print(formatted_response.text)

    return [item.strip().lower() for item in formatted_response.text.split(";")]


# add text here
text = "The Relationship of Urinary Metabolites of Carbaryl/Naphthalene and Chlorpyrifos with Human Semen QualityMost of the general population is exposed to carbaryl and other contemporary-use insecticides at low levels. Studies of laboratory animals, in addition to limited human data, show an association between carbaryl exposure and decreased semen quality. In the present study we explored whether environmental exposures to 1-naphthol (1N), a metabolite of carbaryl and naphthalene, and 3,5,6-trichloro-2-pyridinol (TCPY), a metabolite of chlorpyrifos and chlorpyrifos-methyl, are associated with decreased semen quality in humans. Subjects (n = 272) were recruited through a Massachusetts infertility clinic. Individual exposures were measured as spot urinary concentrations of 1N and TCPY adjusted using specific gravity. Semen quality was assessed as sperm concentration, percent motile sperm, and percent sperm with normal morphology, along with sperm motion parameters (straight-line velocity, curvilinear velocity, and linearity). Median TCPY and 1N concentrations were 3.22 and 3.19 \u03bcg/L, respectively. For increasing 1N tertiles, adjusted odds ratios (ORs) were significantly elevated for below-reference sperm concentration (OR for low, medium, and high tertiles = 1.0, 4.2, 4.2, respectively; p-value for trend = 0.01) and percent motile sperm (1.0, 2.5, 2.4; p-value for trend = 0.01). The sperm motion parameter most strongly associated with 1N was straight-line velocity. There were suggestive, borderline-significant associations for TCPY with sperm concentration and motility, whereas sperm morphology was weakly and nonsignificantly associated with both TCPY and 1N. The observed associations between altered semen quality and 1N are consistent with previous studies of carbaryl exposure, although suggestive associations with TCPY are difficult to interpret because human and animal data are currently limited.Despite the ubiquitous use of insecticides and subsequent exposure among the general population [Centers for Disease Control and Prevention (CDC) 2003; Hill et al. 1995; MacIntosh et al. 1999], there are limited human studies investigating associations between exposure to contemporary-use insecticides at environmental levels and male reproductive health. Human and animal data suggest a potential association between exposures to some commonly used insecticides and decreased semen quality. A study of workers that packaged carbaryl found an increased proportion of oligozoospermic (< 20 million sperm/mL) and teratospermic (> 60% abnormal sperm morphology) men compared with a reference group of chemical workers (Whorton et al. 1979; Wyrobek et al. 1981). Further support for carbaryl\u2019s testicular toxicity comes from studies in laboratory rats that showed associations between carbaryl exposure and sperm shape abnormalities and chromosomal aberrations (Luca and Balan 1987), as well as dose\u2013response relationships between carbaryl exposure and a decline in epididymal sperm count and motility and increased abnormal sperm morphology (Pant et al. 1995, 1996; Rybakova 1966; Shtenberg and Rybakova 1968). Carbaryl was also found to disrupt endocrine regulation of gonadal function in fish (Ghosh and Bhattacharya 1990). Chlorpyrifos, a frequently used insecticide until being banned for residential use (Lewis 2000), is less studied than is carbaryl for its testicular toxicity but has been found to disrupt endocrine regulation in ewes (Rawlings et al. 1998). Recently, the CDC reported measurable levels of urinary 3,5,6-trichloro-2-pyridinol (TCPY), a metabolite of chlorpyrifos and chlorpyrifos-methyl, and 1-naphthol (1N), a metabolite of carbaryl and naphthalene, in > 90% and 75% of males in the United States, respectively (CDC 2003)."
output = getGeminiResult(text)
print(output)


exit(0)

# Specify the path to your CSV file
csv_file_path = "D:\\workspace\\TestingNERTools\\evaluation_data\\nlmchem\\op_new11.csv"

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
    i = 0
    for row in csv_reader:
        i += 1
        if i < 10:
            continue
        print("--------------------->")
        falsePositives = 0
        falseNegatives = 0
        truePositives = 0
        trueNegatives = 0
        perfectMatches = 0
        partialMatches = 0

        text = row["Text"]
        entities = []
        if text.count(" ") > 100:
            no_of_partitions = int(text.count(" ") / 500)
            startIndex = 0
            for i in range(no_of_partitions):
                endIndex = (
                    text.find(". ", int((i + 1) * (len(text) / no_of_partitions)))
                    if int((i + 1) * (len(text) / no_of_partitions)) < len(text)
                    else len(text)
                )
                partialText = text[startIndex:endIndex]
                entities.extend(getGeminiResult(text))
                startIndex = endIndex
        else:
            entities = getGeminiResult(text)
        # You can access specific columns using the column names, e.g., row['ColumnName']
        goldenEntities = set(
            [item.strip().lower() for item in row["Chemical Entities"].split("; ")]
        )
        for entity in set(entities):
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
        print("Gemini: ", set(entities))
        print(
            "NLMChem: ",
            set(
                [item.strip().lower() for item in row["Chemical Entities"].split("; ")]
            ),
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
