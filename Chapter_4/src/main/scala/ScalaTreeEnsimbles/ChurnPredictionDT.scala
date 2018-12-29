package ScalaTreeEnsimbles

import org.apache.spark._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.SQLImplicits
import org.apache.spark.sql._
import org.apache.spark.sql.Dataset
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{ DecisionTreeClassifier, DecisionTreeClassificationModel }
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{ ParamGridBuilder, CrossValidator }

object ChurnPredictionDT {
  def main(args: Array[String]) {
    val spark = SparkSession
      .builder
      .master("local[*]")
      .config("spark.sql.warehouse.dir", "E:/Exp/")
      .appName("ChurnPrediction")
      .getOrCreate()

    import spark.implicits._

    val dTree = new DecisionTreeClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setSeed(12357L)

    // Chain indexers and tree in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(ScalaClassification.PipelineConstruction.ipindexer,
        ScalaClassification.PipelineConstruction.labelindexer,
        ScalaClassification.PipelineConstruction.assembler,
        dTree))

    // Search through decision tree's maxDepth parameter for best model
    var paramGrid = new ParamGridBuilder()
      .addGrid(dTree.impurity, "gini" :: "entropy" :: Nil)
      .addGrid(dTree.maxBins, 3 :: 5 :: 9 :: 10 :: Nil)
      .addGrid(dTree.maxDepth, 5 :: 10 :: 15 :: Nil)
      .build()

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("prediction")

    // Set up 10-fold cross validation
    val numFolds = 10
    val crossval = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(numFolds)

    val cvModel = crossval.fit(ScalaClassification.Preprocessing.trainDF)

    val bestModel = cvModel.bestModel
    println("The Best Model and Parameters:\n--------------------")
    println(bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel].stages(3))

    bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel]
      .stages(3)
      .extractParamMap

    val treeModel = bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel]
      .stages(3)
      .asInstanceOf[DecisionTreeClassificationModel]

    println("Learned classification tree model:\n" + treeModel.toDebugString)
    println("Feature 11:" + ScalaClassification.Preprocessing.trainDF.select(ScalaClassification.PipelineConstruction.featureCols(11)))
    println("Feature 3:" + ScalaClassification.Preprocessing.trainDF.select(ScalaClassification.PipelineConstruction.featureCols(3)))

    val predictions = cvModel.transform(ScalaClassification.Preprocessing.testSet)
    predictions.show(10)

    val result = predictions.select("label", "prediction", "probability")
    val resutDF = result.withColumnRenamed("prediction", "Predicted_label")
    resutDF.show(10)

    val accuracy = evaluator.evaluate(predictions)
    println("Accuracy: " + accuracy)
    evaluator.explainParams()

    val predictionAndLabels = predictions
      .select("prediction", "label")
      .rdd.map(x => (x(0).asInstanceOf[Double], x(1)
        .asInstanceOf[Double]))

    val metrics = new BinaryClassificationMetrics(predictionAndLabels)
    val areaUnderPR = metrics.areaUnderPR
    println("Area under the precision-recall curve: " + areaUnderPR)

    val areaUnderROC = metrics.areaUnderROC
    println("Area under the receiver operating characteristic (ROC) curve: " + areaUnderROC)

    /*
    val precesion = metrics.precisionByThreshold()
    println("Precision: "+ precesion.foreach(print))

    val recall = metrics.recallByThreshold()
    println("Recall: "+ recall.foreach(print))

    val f1Measure = metrics.fMeasureByThreshold()
    println("F1 measure: "+ f1Measure.foreach(print))
										 * 
										 */

    val lp = predictions.select("label", "prediction")
    val counttotal = predictions.count()
    val correct = lp.filter($"label" === $"prediction").count()
    val wrong = lp.filter(not($"label" === $"prediction")).count()
    val ratioWrong = wrong.toDouble / counttotal.toDouble
    val ratioCorrect = correct.toDouble / counttotal.toDouble
    val tp = lp.filter($"prediction" === 0.0).filter($"label" === $"prediction").count() / counttotal.toDouble
    val tn = lp.filter($"prediction" === 1.0).filter($"label" === $"prediction").count() / counttotal.toDouble
    val fp = lp.filter($"prediction" === 1.0).filter(not($"label" === $"prediction")).count() / counttotal.toDouble
    val fn = lp.filter($"prediction" === 0.0).filter(not($"label" === $"prediction")).count() / counttotal.toDouble
    
    val MCC = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (fp + tn) * (tn + fn))

    println("Total Count: " + counttotal)
    println("Correct: " + correct)
    println("Wrong: " + wrong)
    println("Ratio wrong: " + ratioWrong)
    println("Ratio correct: " + ratioCorrect)
    println("Ratio true positive: " + tp)
    println("Ratio false positive: " + fp)
    println("Ratio true negative: " + tn)
    println("Ratio false negative: " + fn)
    println("Matthews correlation coefficient: " + MCC)
  }
}