package ScalaClassification

import org.apache.spark._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.classification.{ BinaryLogisticRegressionSummary, LogisticRegression, LogisticRegressionModel }
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.{ ParamGridBuilder, CrossValidator }
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

object ChurnPredictionLR {
  def main(args: Array[String]) {
    val spark = SparkSession
      .builder
      .master("local[*]")
      .config("spark.sql.warehouse.dir", "E:/Exp/")
      .appName("ChurnPrediction")
      .getOrCreate()
    import spark.implicits._

    val numFolds = 10
    val MaxIter: Seq[Int] = Seq(100)
    val RegParam: Seq[Double] = Seq(0.01) // L2 regularization param, set 0.10 with L1 reguarization
    val Tol: Seq[Double] = Seq(1e-4)
    val ElasticNetParam: Seq[Double] = Seq(1.0) // Combination of L1 and L2

    val lr = new LogisticRegression()
      .setLabelCol("label")
      .setFeaturesCol("features")

    // Chain indexers and tree in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(PipelineConstruction.ipindexer,
        PipelineConstruction.labelindexer,
        PipelineConstruction.assembler,
        lr))

    // Search through decision tree's maxDepth parameter for best model                               
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.maxIter, MaxIter)
      .addGrid(lr.regParam, RegParam)
      .addGrid(lr.tol, Tol)
      .addGrid(lr.elasticNetParam, ElasticNetParam)
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