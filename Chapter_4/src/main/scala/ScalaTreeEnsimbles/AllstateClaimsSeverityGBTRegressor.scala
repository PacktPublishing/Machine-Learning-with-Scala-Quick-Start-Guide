package ScalaTreeEnsimbles

import org.apache.spark.ml.regression.{ GBTRegressor, GBTRegressionModel }
import org.apache.spark.ml.{ Pipeline, PipelineModel }
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.log4j.LogManager

object AllstateClaimsSeverityGBTRegressor {
  def main(args: Array[String]) {
    val spark = SparkSession
      .builder
      .master("local[*]")
      .config("spark.sql.warehouse.dir", "E:/Exp/")
      .appName(s"OneVsRestExample")
      .getOrCreate()
    import spark.implicits._

    // Estimator algorithm
    val model = new GBTRegressor().setFeaturesCol("features").setLabelCol("label")

    // Building the Pipeline for transformations and predictor
    val pipeline = new Pipeline().setStages((RegressionAnalysis.Preproessing.stringIndexerStages :+ RegressionAnalysis.Preproessing.assembler) :+ model)

    // ***********************************************************
    println("Preparing K-fold Cross Validation and Grid Search")
    // ***********************************************************

    // Search through decision tree's maxDepth parameter for best model
    var paramGrid = new ParamGridBuilder()
      .addGrid(model.impurity, "variance" :: Nil)// variance for regression
      .addGrid(model.maxBins, 25 :: 30 :: 35 :: Nil)
      .addGrid(model.maxDepth, 5 :: 10 :: 15 :: Nil)
      .build()

    val numFolds = 10  
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new RegressionEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(numFolds)

    // ************************************************************
    println("Training model with GradientBoostedTrees algorithm")
    // ************************************************************

    val cvModel = cv.fit(RegressionAnalysis.Preproessing.trainingData)

    // **********************************************************************
    println("Evaluating model on train and test data and calculating RMSE")
    // **********************************************************************

    val trainPredictionsAndLabels = cvModel.transform(RegressionAnalysis.Preproessing.trainingData).select("label", "prediction")
      .map { case Row(label: Double, prediction: Double) => (label, prediction) }.rdd

    val validPredictionsAndLabels = cvModel.transform(RegressionAnalysis.Preproessing.testData).select("label", "prediction")
      .map { case Row(label: Double, prediction: Double) => (label, prediction) }.rdd

    val trainRegressionMetrics = new RegressionMetrics(trainPredictionsAndLabels)
    val validRegressionMetrics = new RegressionMetrics(validPredictionsAndLabels)

    val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel]
    val featureImportances = bestModel.stages.last.asInstanceOf[GBTRegressionModel].featureImportances.toArray

    val FI_to_List_sorted = featureImportances.toList.sorted.toArray
    //val sortedFI = FI_to_List.sorted    
    print(FI_to_List_sorted)

    val output = "\n=====================================================================\n" +
      s"Param trainSample: ${RegressionAnalysis.Preproessing.trainSample}\n" +
      s"Param testSample: ${RegressionAnalysis.Preproessing.testSample}\n" +
      s"TrainingData count: ${RegressionAnalysis.Preproessing.trainingData.count}\n" +
      s"ValidationData count: ${RegressionAnalysis.Preproessing.testData.count}\n" +
      s"TestData count: ${RegressionAnalysis.Preproessing.testData.count}\n" +
      "=====================================================================\n" +
      s"Training data MSE = ${trainRegressionMetrics.meanSquaredError}\n" +
      s"Training data RMSE = ${trainRegressionMetrics.rootMeanSquaredError}\n" +
      s"Training data R-squared = ${trainRegressionMetrics.r2}\n" +
      s"Training data MAE = ${trainRegressionMetrics.meanAbsoluteError}\n" +
      s"Training data Explained variance = ${trainRegressionMetrics.explainedVariance}\n" +
      "=====================================================================\n" +
      s"Validation data MSE = ${validRegressionMetrics.meanSquaredError}\n" +
      s"Validation data RMSE = ${validRegressionMetrics.rootMeanSquaredError}\n" +
      s"Validation data R-squared = ${validRegressionMetrics.r2}\n" +
      s"Validation data MAE = ${validRegressionMetrics.meanAbsoluteError}\n" +
      s"Validation data Explained variance = ${validRegressionMetrics.explainedVariance}\n" +
      "=====================================================================\n" +
      s"CV params explained: ${cvModel.explainParams}\n" +
      s"GBT params explained: ${bestModel.stages.last.asInstanceOf[GBTRegressionModel].explainParams}\n" +
      s"GBT features importances:\n ${RegressionAnalysis.Preproessing.featureCols.zip(FI_to_List_sorted).map(t => s"\t${t._1} = ${t._2}").mkString("\n")}\n" +
      "=====================================================================\n"

    println(output)

    // *****************************************
    println("Run prediction over test dataset")
    // *****************************************

    // Predicts and saves file ready for Kaggle!
    //if(!params.outputFile.isEmpty){
    cvModel.transform(RegressionAnalysis.Preproessing.testData)
      .select("id", "prediction")
      .withColumnRenamed("prediction", "loss")
      .coalesce(1)
      .write.format("com.databricks.spark.csv")
      .option("header", "true")
      .save("output/result_GBT.csv")
    //}
  }
}