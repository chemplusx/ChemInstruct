package data;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.lang3.tuple.Pair;

import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class CustomDataset  implements BaseDataset{

    public HashMap<String, Pair<List<String>, String>> dataToMatch = new HashMap<>();

    public HashMap<String, Pair<List<String>, String>> GetDatasetForEvaluation(){

        return new HashMap<>();
    }

    private String inputFile;
    public void SetInputFile(String directory){
        this.inputFile = directory + "custom\\TestNERDataset.csv";
    }

    public HashMap<String, Pair<List<String>, String>> getDataToMatch() {
        return dataToMatch;
    }

    @Override
    public void BuildEvaluationData() throws Exception {
        try {
            FileReader fileReader = new FileReader(this.inputFile);
            CSVParser parser = new CSVParser(fileReader, CSVFormat.DEFAULT.withFirstRecordAsHeader());
            List<Map<String, String>> records = new ArrayList<>();
            int k=0;
            for (CSVRecord record : parser) {
                String chemicals = record.toMap().get("chemicals");
                if(Objects.isNull(chemicals) || (chemicals.isEmpty() || chemicals.isBlank())){
                    continue;
                }
                List<String> entities = new ArrayList<>(Arrays.asList(record.toMap().get("chemicals").split(", ")));
                for(int i=0;i<entities.size();i++){
                    entities.set(i, entities.get(i).toLowerCase().replaceAll("derivatives", ""));
                    entities.set(i, entities.get(i).toLowerCase().replaceAll("derivative", ""));
                }
                Set<String> uniq = new HashSet<>(entities);
                List<String> ee = new ArrayList<String>(uniq);
                this.dataToMatch.put("custom_"+k, Pair.of(ee, record.toMap().get("text")));
                k++;
//                this.dataToMatch.put(record.toMap().get("Open access "), entities);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
