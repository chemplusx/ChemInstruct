package tools.utils;

import data.BaseDataset;
import data.ChemDNer;
import data.CustomDataset;
import data.NLMChem;
import models.DatasetType;
import models.EvalMetric;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.text.similarity.LevenshteinDistance;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

public class MatchingUtils {

    private BaseDataset datasetForEvaluation;
    private String evaluationDumpFile = "evalDump.csv";

    public MatchingUtils(){}


    public BaseDataset getDatasetForEvaluation() {
        return datasetForEvaluation;
    }

    public void setDatasetForEvaluation(BaseDataset datasetForEvaluation) {
        this.datasetForEvaluation = datasetForEvaluation;
    }

    private String escapeCsvValue(String value) {
        // If the value contains commas, enclose it within double quotes
        if (value.contains(",")) {
            return "\"" + value + "\"";
        } else {
            return value;
        }
    }

    public void createEvalDumpFile(String toolName){
        try (FileWriter writer = new FileWriter(toolName+this.evaluationDumpFile)) {
            // Loop through the list and write each string as a single column in the CSV file
            writer.append("NLMChem");
            writer.append(",");
            writer.append(toolName);
            writer.append("\n");

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    private void writeCsvFile( List<String> dataList1, List<String> dataList2, String toolName) {
        try (FileWriter writer = new FileWriter(toolName + this.evaluationDumpFile, true)) {
            // Loop through the list and write each string as a single column in the CSV file
            writer.append(this.escapeCsvValue(String.join(";", dataList1)));
            writer.append(",");
            writer.append(this.escapeCsvValue(String.join(";", dataList2)));
            writer.append("\n");

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void InitialiseDataset(DatasetType dataset) throws Exception {
        switch (dataset) {
            case NLMCHEM:
                this.datasetForEvaluation = new NLMChem();
                break;
            case CHEMDNER:
                this.datasetForEvaluation = new ChemDNer();
                break;
            case CUSTOM:
                this.datasetForEvaluation = new CustomDataset();
                break;
            default:
                System.out.println("Unknown Dataset name");
                throw new Exception("Unknown dataset name");
        }
    }

    public void BuildEvaluationData(DatasetType dataset) throws Exception {
        this.datasetForEvaluation.BuildEvaluationData();
    }

    public void BuildEvaluationDataFromNLMChem() {
        try {
            String csvFilePath = "D:\\workspace\\TestingNERTools\\evaluation_data\\input.csv";
            FileReader fileReader = new FileReader(csvFilePath);
            CSVParser parser = new CSVParser(fileReader, CSVFormat.DEFAULT.withFirstRecordAsHeader());
            List<Map<String, String>> records = new ArrayList<>();
            int index=0;
            for (CSVRecord record : parser) {
                List<String> entities = new ArrayList<>(Arrays.asList(record.toMap().get("Chemical Entities").split("; ")));
                for(int i=0;i<entities.size();i++){
                    entities.set(i, entities.get(i).toLowerCase().replaceAll("derivatives", ""));
                    entities.set(i, entities.get(i).toLowerCase().replaceAll("derivative", ""));
                }
                Set<String> uniq = new HashSet<>(entities);
                List<String> ee = new ArrayList<String>(uniq);
                String identifier = "id"+index++;
                this.datasetForEvaluation.getDataToMatch().put(identifier, Pair.of(ee, record.toMap().get("Text")));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public double match(String str1, String str2){
        LevenshteinDistance levenshteinDistance = new LevenshteinDistance();
        int distance = levenshteinDistance.apply(str1.toLowerCase(), str2.toLowerCase());

        int starti = str2.indexOf(" (");
        int endi = str2.indexOf(")", starti);

        if(starti!=-1 && endi!=-1){
            String str22 = str2.substring(0, starti) + str2.substring(endi+1);
            int distance2 = levenshteinDistance.apply(str1.toLowerCase(), str2.toLowerCase());
//            System.out.println("Levenshtein Distance: "+str1+", "+str22 +" : " + Math.min(distance, distance2));
            return Math.min(distance, distance2);
        }

//        System.out.println("Levenshtein Distance: "+str1+", "+str2 +" : " + distance);

        return distance;
        // Decide if they are similar based on a threshold
        // int threshold = 2; // Example threshold
        // if (distance <= threshold) {
        //     return 
        // } else {
        //     System.out.println("Chemical names are not similar.");
        // }
    }


    public EvalMetric matchAndConnect(List<String> collectedData, String identifier, String tool){
        EvalMetric em = new EvalMetric();
        List<String> copiedSourceData = new ArrayList<>(this.datasetForEvaluation.getDataToMatch().get(identifier).getLeft());
        this.writeCsvFile(copiedSourceData, collectedData, tool);
        for(String text: collectedData){
            double minLd = 10000000;
            String texte = "";
            for(String sText: this.datasetForEvaluation.getDataToMatch().get(identifier).getLeft()){
                double ld = this.match(text, sText);
                if(minLd > ld){
                    texte = sText;
                    minLd = ld;
                }
            }
            if(minLd <= 2){
                em.perfectMatches++;
                em.truePositives++;
                copiedSourceData.remove(texte);
            }
        }
        em.falsePositives += collectedData.size() - em.truePositives;
        em.falseNegatives += copiedSourceData.size();
        return em;
    }
}