package data;

import org.apache.commons.lang3.tuple.Pair;

import java.util.HashMap;
import java.util.List;

public interface BaseDataset {
    HashMap<String, Pair<List<String>, String>> GetDatasetForEvaluation();
    void BuildEvaluationData() throws Exception;

    void SetInputFile(String directory);

    HashMap<String, Pair<List<String>, String>> getDataToMatch();
}
