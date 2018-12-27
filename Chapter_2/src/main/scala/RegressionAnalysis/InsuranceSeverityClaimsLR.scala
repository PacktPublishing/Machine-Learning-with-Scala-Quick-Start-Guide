package RegressionAnalysis

import org.apache.spark.ml.regression.{LinearRegression,LinearRegressionModel}
import org.apache.spark.ml.{ Pipeline, PipelineModel }
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.evaluation.RegressionMetrics

import org.apache.log4j.Logger
import org.apache.log4j.Level

object InsuranceSeverityClaimsLR {
  def main(args: Array[String]) {
    val spark = SparkSession
      .builder
      .master("local[*]")
      .config("spark.sql.warehouse.dir", "E:/Exp/")
      .appName(s"OneVsRestExample")
      .getOrCreate()

    import spark.implicits._

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    // Create an LinerRegression estimator
    val lr = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")

    // Building the Pipeline model for transformations and predictor
    println("Building ML pipeline")
    val pipeline = new Pipeline().setStages((Preproessing.stringIndexerStages :+ Preproessing.assembler) :+ lr)
    val glrPipelineModel = pipeline.fit(Preproessing.trainingData)

    // **********************************************************************
    println("Evaluating the model on the training and validation set and calculating RMSE")
    // **********************************************************************
    val trainPredictionsAndLabels = glrPipelineModel.transform(Preproessing.trainingData).select("label", "prediction")
      .map { case Row(label: Double, prediction: Double) => (label, prediction) }.rdd

    val validPredictionsAndLabels = glrPipelineModel.transform(Preproessing.validData).select("label", "prediction")
      .map { case Row(label: Double, prediction: Double) => (label, prediction) }.rdd
    
    val trainRegressionMetrics = new RegressionMetrics(trainPredictionsAndLabels)
    val validRegressionMetrics = new RegressionMetrics(validPredictionsAndLabels)

    val results = "\n=====================================================================\n" +
      s"Param trainSample: ${Preproessing.trainSample}\n" +
      s"Param testSample: ${Preproessing.testSample}\n" +
      s"TrainingData count: ${Preproessing.trainingData.count}\n" +
      s"ValidationData count: ${Preproessing.testData.count}\n" +
      s"TestData count: ${Preproessing.testData.count}\n" +
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
      "=====================================================================\n"
    println(results)

    /*
    // ***********************************************************
    println("Preparing K-fold Cross Validation and Grid Search: Model tuning")
    // ***********************************************************
    val paramGrid = new ParamGridBuilder()
      .addGrid(glr.maxIter, Array(10, 20, 30))
      .addGrid(glr.regParam, Array(0.1, 0.2, 0.3))
      .addGrid(glr.tol, Array(0.001, 0.01, 0.1))
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new RegressionEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(numFolds)

    // ************************************************************
    println("Training model with Linear Regression algorithm")
    // ************************************************************
    val cvModel = cv.fit(Preproessing.trainingData) 
    //val cvModel2 = cv.fit(Preproessing.testData)

    // Save the workflow
    cvModel.write.overwrite().save("model/AFT_model")

    // Load the workflow back
    val sameCV = CrossValidatorModel.load("model/AFT_model")

    // **********************************************************************
    println("Evaluating model on the validation set and calculating RMSE")
    // **********************************************************************
    val validPredictionsAndLabels = cvModel.transform(Preproessing.validationData).select("label", "prediction")
      .map { case Row(label: Double, prediction: Double) => (label, prediction) }.rdd

    val validRegressionMetrics = new RegressionMetrics(validPredictionsAndLabels)
    val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel]

    val results = "\n=====================================================================\n" +
      s"Param trainSample: ${Preproessing.trainSample}\n" +
      s"Param testSample: ${Preproessing.testSample}\n" +
      s"TrainingData count: ${Preproessing.trainingData.count}\n" +
      s"ValidationData count: ${Preproessing.validationData.count}\n" +
      s"TestData count: ${Preproessing.testData.count}\n" +
      "=====================================================================\n" +
      s"Param numFolds = ${numFolds}\n" +
      "=====================================================================\n" +
      s"Validation data MSE = ${validRegressionMetrics.meanSquaredError}\n" +
      s"Validation data RMSE = ${validRegressionMetrics.rootMeanSquaredError}\n" +
      s"Validation data R-squared = ${validRegressionMetrics.r2}\n" +
      s"Validation data MAE = ${validRegressionMetrics.meanAbsoluteError}\n" +
      s"Validation data Explained variance = ${validRegressionMetrics.explainedVariance}\n" +
      "=====================================================================\n"
    println(results)
    * 
    */

    // *****************************************
    println("Run prediction on the test set")
    glrPipelineModel.transform(Preproessing.testData)
      .select("id", "prediction")
      .withColumnRenamed("prediction", "loss")
      .coalesce(1)
      .write.format("com.databricks.spark.csv")
      .option("header", "true")
      .save("output/result_GLR.csv")

    spark.stop()
  }
}