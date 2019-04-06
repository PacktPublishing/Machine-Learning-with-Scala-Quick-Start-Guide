package ScalaTreeEnsimbles

import org.apache.spark._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{ RandomForestClassifier, RandomForestClassificationModel }
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{ ParamGridBuilder, CrossValidator }

object ChurnPredictionRF {
  def main(args: Array[String]) {
    val spark = SparkSession
      .builder
      .master("local[*]")
      .config("spark.sql.warehouse.dir", "E:/Exp/")
      .appName("ChurnPrediction")
      .getOrCreate()

    import spark.implicits._

    val rf = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setSeed(1234567L)

    // Chain indexers and tree in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(ScalaClassification.PipelineConstruction.ipindexer,
        ScalaClassification.PipelineConstruction.labelindexer,
        ScalaClassification.PipelineConstruction.assembler,
        rf))

    // Search through decision tree's maxDepth parameter for best model
    val paramGrid = new ParamGridBuilder()
      .addGrid(rf.maxDepth, 3 :: 5 :: 10 :: Nil) // :: 15 :: 20 :: 25 :: 30 :: Nil)
      .addGrid(rf.featureSubsetStrategy, "auto" :: "all" :: Nil)
      .addGrid(rf.impurity, "gini" :: "entropy" :: Nil)
      .addGrid(rf.maxBins, 5 :: 10 :: 20 :: Nil) //10 :: 15 :: 25 :: 35 :: 45 :: Nil)
      .addGrid(rf.numTrees, 5 :: 10 :: 20 :: Nil) // :: 100 :: Nil)
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

    // Save the workflow
    cvModel.write.overwrite().save("model/RF_model_churn")

    val bestModel = cvModel.bestModel
    println("The Best Model and Parameters:\n--------------------")
    println(bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel].stages(3))

    bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel]
      .stages(3)
      .extractParamMap

    val treeModel = bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel]
      .stages(3)
      .asInstanceOf[RandomForestClassificationModel]

    println("Learned classification tree model:\n" + treeModel.toDebugString)
    println("Feature 11:" + ScalaClassification.Preprocessing.trainDF.select(ScalaClassification.PipelineConstruction.featureCols(11)))
    println("Feature 3:" + ScalaClassification.Preprocessing.trainDF.select(ScalaClassification.PipelineConstruction.featureCols(3)))

    val predDF = cvModel.transform(ScalaClassification.Preprocessing.testSet)
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