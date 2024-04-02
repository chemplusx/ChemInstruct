package data;

import org.apache.commons.lang3.tuple.Pair;

import java.util.HashMap;
import java.util.List;

public class ChemDNer implements BaseDataset{

    public HashMap<String, Pair<List<String>, String>> dataToMatch = new HashMap<>();

    public HashMap<String, Pair<List<String>, String>> GetDatasetForEvaluation(){

        return new HashMap<>();
    }

    private String inputFile;
    public void SetInputFile(String directory){
        this.inputFile = directory + "chemdner\\input.csv";
    }

    public HashMap<String, Pair<List<String>, String>> getDataToMatch() {
        return dataToMatch;
    }

    public void BuildEvaluationData(){

    }
}
