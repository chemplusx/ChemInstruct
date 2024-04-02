import java.util.HashMap;

import data.BaseDataset;
import models.DatasetType;
import models.Tools;

import tools.CheNERTool;
import tools.ChemDETool;
import tools.ChemspotTool;
import tools.utils.ArgumentParser;

public class StartEvaluation {
    
    public static void main(String[] args) throws Exception {
    System.out.println("Hello World"+ System.getProperty("user.dir"));
        if(args.length > 0) {
            HashMap<String, String> cliArgs = (new ArgumentParser()).ParseCliArguments(args);

            RunEvaluation(cliArgs.get("directory"), cliArgs.get("dataset"), cliArgs.get("tool"));
        }else{
            String directoryPath = System.getProperty("user.dir") + "\\dataset\\";
            String defaultDataset = "custom", defaultToolToTest = "chener";
            RunEvaluation(directoryPath, defaultDataset, defaultToolToTest);
        }
    }

    public static void RunEvaluation(String directoryPath,String dataset, String toolName) throws Exception {
        BaseDataset dfe;
        switch (Tools.ValueOf(toolName)) {
            case CHEMSPOT -> {
                ChemspotTool testchemspot = new ChemspotTool();
                testchemspot.matchingUtils.InitialiseDataset(DatasetType.ValueOf(dataset));
                dfe = testchemspot.matchingUtils.getDatasetForEvaluation();
                dfe.SetInputFile(directoryPath);
                testchemspot.matchingUtils.setDatasetForEvaluation(dfe);
                testchemspot.BuildDataForTesting(true, dataset);
                testchemspot.RunToolAndEvaluate("Chemspot");
            }
            case CHENER -> {
                CheNERTool testchener = new CheNERTool();
                testchener.matchingUtils.InitialiseDataset(DatasetType.ValueOf(dataset));
                dfe = testchener.matchingUtils.getDatasetForEvaluation();
                dfe.SetInputFile(directoryPath);
                testchener.matchingUtils.setDatasetForEvaluation(dfe);
                testchener.BuildDataForTesting(true, dataset);
                testchener.RunToolAndEvaluate("CheNER");
            }
            case CHEMICAL_DATA_EXTRACTOR -> {
                ChemDETool testchemde = new ChemDETool();
                testchemde.matchingUtils.InitialiseDataset(DatasetType.ValueOf(dataset));
                dfe = testchemde.matchingUtils.getDatasetForEvaluation();
                dfe.SetInputFile(directoryPath);
                testchemde.matchingUtils.setDatasetForEvaluation(dfe);
                testchemde.BuildDataForTesting(true, dataset);
                testchemde.RunToolAndEvaluate("ChemDataExtractor");
            }
        }
    }

}