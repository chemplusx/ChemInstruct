package tools;

import models.DatasetType;
import models.EvalMetric;
import tools.utils.MatchingUtils;
import tools.utils.ScoringUtils;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public abstract class BaseTool {
    public String outputFilePath;
    public Boolean evaluate = false;

    public ScoringUtils scoringUtils = new ScoringUtils();
    public MatchingUtils matchingUtils = new MatchingUtils();

    public List<String> ExtractEntities(String text) throws Exception {
        throw new Exception("Method not implemented");
    }

    public void BuildDataForTesting(Boolean evalMode, String datasetName) throws Exception {
        if(evalMode){
            this.evaluate = true;
            this.matchingUtils.BuildEvaluationData(DatasetType.ValueOf(datasetName));
//            this.matchingUtils.BuildEvaluationDataFromNLMChem();
        }
    }

    public void RunToolAndEvaluate(String toolName) throws Exception {
        this.matchingUtils.createEvalDumpFile(toolName);
        for(String id: this.matchingUtils.getDatasetForEvaluation().getDataToMatch().keySet()) {
            if(this.matchingUtils.getDatasetForEvaluation().getDataToMatch().get(id).getLeft().isEmpty()){
                continue;
            }

            String tt = this.matchingUtils.getDatasetForEvaluation().getDataToMatch().get(id).getRight();
            List<String> entitiesExtracted = this.ExtractEntities(tt);

            if (this.evaluate) {
                Set<String> uniq = new HashSet<>(entitiesExtracted);
                List<String> ee = new ArrayList<>(uniq);
                EvalMetric em = this.matchingUtils.matchAndConnect(ee, id, toolName);
                this.scoringUtils.metric.falseNegatives += em.falseNegatives;
                this.scoringUtils.metric.falsePositives += em.falsePositives;
                this.scoringUtils.metric.trueNegatives += em.trueNegatives;
                this.scoringUtils.metric.truePositives += em.truePositives;
                this.scoringUtils.metric.perfectMatches += em.perfectMatches;
                this.scoringUtils.metric.partialMatches += em.partialMatches;
                System.out.println(id);
                System.out.println("This Iteration:");
                ScoringUtils ls = new ScoringUtils();
                ls.metric = em;
                ls.evaluateAll();
                System.out.println("Total:");
                this.scoringUtils.evaluateAll();
            }
        }
        this.scoringUtils.evaluateAll();
    }
}
