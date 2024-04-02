package tools.utils;

import models.EvalMetric;

public class ScoringUtils {

    public EvalMetric metric;

    public ScoringUtils() {
        this.metric = new EvalMetric();
    }

    public double precision() {
        return this.metric.truePositives / (double) (this.metric.truePositives + this.metric.falsePositives);
    }

    // Calculates Recall
    public double recall() {
        return this.metric.truePositives / (double) (this.metric.truePositives + this.metric.falseNegatives);
    }

    // Calculates F1 Score
    public double f1Score() {
        double prec = this.precision();
        double rec = this.recall();
        return 2 * (prec * rec) / (prec + rec);
    }

    public double r2Score(double[] actual, double[] predictions) {
        double totalSumOfSquares = 0;
        double residualSumOfSquares = 0;
        double meanActual = 0;

        for (double a : actual) {
            meanActual += a;
        }
        meanActual /= actual.length;

        for (int i = 0; i < actual.length; i++) {
            totalSumOfSquares += Math.pow((actual[i] - meanActual), 2);
            residualSumOfSquares += Math.pow((actual[i] - predictions[i]), 2);
        }

        return 1 - (residualSumOfSquares / totalSumOfSquares);
    }

    public double evaluateAll() {
        System.out.println("Precision: " + this.precision());
        System.out.println("Recall: " + this.recall());
        System.out.println("F1Score: " + this.f1Score());
//        this.r2Score();
        return 0.0;
    }
}
