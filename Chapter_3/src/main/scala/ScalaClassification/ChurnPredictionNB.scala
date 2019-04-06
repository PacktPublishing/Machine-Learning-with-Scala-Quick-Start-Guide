package ScalaClassification

import org.apache.spark._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.classification.{ BinaryLogisticRegressionSummary, NaiveBayes, NaiveBayesModel }
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.{ ParamGridBuilder, CrossValidator }
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}

/*
class Stats(val tp: Int, val tn: Int, val fp: Int, val fn: Int) {
  val TPR = tp / (tp + fn).toDouble
  val recall = TPR
  val sensitivity = TPR
  val TNR = tn / (tn + fp).toDouble
  val specificity = TNR
  val PPV = tp / (tp + fp).toDouble
  val precision = PPV
  val NPV = tn / (tn + fn).toDouble
  val FPR = 1.0 - specificity
  val FNR = 1.0 - recall
  val FDR = 1.0 - precision
  val ACC = (tp + tn) / (tp + fp + fn + tn).toDouble
  val accuracy = ACC
  val F1 = 2 * PPV * TPR / (PPV + TPR).toDouble
  val MCC = (tp * tn - fp * fn).toDouble / math.sqrt((tp + fp).toDouble * (tp + fn).toDouble * (fp + tn).toDouble * (tn + fn).toDouble)
} */

object ChurnPredictionNB {
  def main(args: Array[String]) {
    val spark = SparkSession
      .builder
      .master("local[*]")
      .config("spark.sql.warehouse.dir", "E:/Exp/")
      .appName("ChurnPrediction")
      .getOrCreate()

    import spark.implicits._

    val numFolds = 10
    val nb = new NaiveBayes()
      .setLabelCol("label")
      .setFeaturesCol("features")

    // Chain indexers and tree in a Pipeline.
    val pipeline = new Pipeline().setStages(Array(PipelineConstruction.ipindexer,
                                                  PipelineConstruction.labelindexer,
                                                  PipelineConstruction.assembler,
                                                  nb))

    // Search through Naive Bayes's smoothing parameter for best model                               
    val paramGrid = new ParamGridBuilder()
      .addGrid(nb.smoothing, Array(1e-2, 1e-4, 1e-6, 1e-8))
      .build()

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("prediction")

    // Set up 10-fold cross validation
    val crossval = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(numFolds)

    val cvModel = crossval.fit(Preprocessing.trainDF)

    val predDF = cvModel.transform(Preprocessing.testSet)
    val result = predDF.select("label", "prediction", "probability")
    val resutDF = result.withColumnRenamed("prediction", "Predicted_label")
    resutDF.show(10)

    val accuracy = evaluator.evaluate(predDF)
    println("Classification accuracy: " + accuracy)

    // Compute other performence metrices
    val predictionAndLabels = predDF
      .select("prediction", "label")
      .rdd.map(x => (x(0).asInstanceOf[Double], x(1)
        .asInstanceOf[Double]))

    val metrics = new BinaryClassificationMetrics(predictionAndLabels)
    val areaUnderPR = metrics.areaUnderPR
    println("Area under the precision-recall curve: " + areaUnderPR)

    val areaUnderROC = metrics.areaUnderROC
    println("Area under the receiver operating characteristic (ROC) curve: " + areaUnderROC)

    val tVSpDF = predDF.select("label", "prediction") // True vs predicted labels
    val TC = predDF.count() //Total count

    val tp = tVSpDF.filter($"prediction" === 0.0).filter($"label" === $"prediction").count() / TC.toDouble
    val tn = tVSpDF.filter($"prediction" === 1.0).filter($"label" === $"prediction").count() / TC.toDouble
    val fp = tVSpDF.filter($"prediction" === 1.0).filter(not($"label" === $"prediction")).count() / TC.toDouble
    val fn = tVSpDF.filter($"prediction" === 0.0).filter(not($"label" === $"prediction")).count() / TC.toDouble
    
    val MCC = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (fp + tn) * (tn + fn)) // Calculating Matthews correlation coefficient

    println("True positive rate: " + tp *100 + "%")
    println("False positive rate: " + fp * 100 + "%")
    println("True negative rate: " + tn * 100 + "%")
    println("False negative rate: " + fn * 100 + "%")
    println("Matthews correlation coefficient: " + MCC)
  }
}