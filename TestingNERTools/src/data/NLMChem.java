package data;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.lang3.tuple.Pair;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class NLMChem  implements BaseDataset{

    public HashMap<String, Pair<List<String>, String>> dataToMatch = new HashMap<>();

    public HashMap<String, Pair<List<String>, String>> GetDatasetForEvaluation(){

        return new HashMap<>();
    }

    private String inputFile;
    private String inputFileJson;

    public void SetInputFile(String directory){
        this.inputFileJson = directory + "nlmchem\\input.json";
        this.inputFile = directory + "nlmchem\\op_new11.csv";
    }

    public HashMap<String, Pair<List<String>, String>> getDataToMatch() {
        return dataToMatch;
    }



    public void BuildEvaluationData() {
        this.BuildEvaluationDataFromJson();
        try {
            FileReader fileReader = new FileReader(this.inputFile);
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
                this.dataToMatch.put(identifier, Pair.of(ee, record.toMap().get("Text")));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void BuildEvaluationDataFromJson() {
        try {
            ObjectMapper objectMapper = new ObjectMapper();
            JsonNode node = objectMapper.readTree(new File(this.inputFileJson));

            if(node.isObject()) {
                Iterator<Map.Entry<String, JsonNode>> fieldsIterator = node.fields();
                while (fieldsIterator.hasNext()) {
                    Map.Entry<String, JsonNode> field = fieldsIterator.next();
                    String identifier = field.getKey();
                    JsonNode value = field.getValue();
                    String text = value.get("text").asText();
                    HashSet<String> entitiesSet = new HashSet<>();
                    for (JsonNode oNode : value.get("entities")) {
                        entitiesSet.add(oNode.asText().toLowerCase());
                    }

                    this.dataToMatch.put(identifier, Pair.of(new ArrayList<>(entitiesSet), text));
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
