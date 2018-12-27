package RegressionAnalysis

import org.apache.spark.ml.regression.{ GeneralizedLinearRegression, GeneralizedLinearRegressionModel }
import org.apache.spark.ml.{ Pipeline, PipelineModel }
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.evaluation.RegressionMetrics

import org.apache.log4j.Logger
import org.apache.log4j.Level

object GeneralizedLinearRegressor {
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
    val glr = new GeneralizedLinearRegression()
      .setFamily("gaussian")// continuous values being predicted
      .setLink("identity")
      .setFeaturesCol("features")
      .setLabelCol("label")

    // Building the Pipeline model for transformations and predictor
    println("Building ML pipeline")
    Preproessing.trainingData.show(5)
    val pipeline = new Pipeline().setStages((Preproessing.stringIndexerStages :+ Preproessing.assembler) :+ glr)
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

    // *****************************************
    println("Run prediction on the test set")
    glrPipelineModel.transform(Preproessing.testData)
      .select("id", "prediction")
      .withColumnRenamed("prediction", "loss")
      .coalesce(1)
      .write.format("com.databricks.spark.csv")
      .option("header", "true")
      //.save("output/result_GLR_latest.csv")    

    // ***********************************************************
    println("Preparing K-fold Cross Validation and Grid Search: Model tuning")
    // ***********************************************************
    val paramGrid = new ParamGridBuilder()
      .addGrid(glr.maxIter, Array(10, 20, 30))
      .addGrid(glr.regParam, Array(0.1, 0.2))
      .addGrid(glr.tol, Array(0.1, 0.2))
      .build()

    val numFolds = 10 //10-fold cross-validation
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
    //cvModel.write.overwrite().save("model/GLR_model")

    // Load the workflow back
    //val sameCV = CrossValidatorModel.load("model/GLR_model")

    // **********************************************************************
    println("Evaluating model on the validation set and calculating RMSE")
    // **********************************************************************
    val trainPredictionsAndLabelsCV = cvModel.transform(Preproessing.trainingData).select("label", "prediction")
      .map { case Row(label: Double, prediction: Double) => (label, prediction) }.rdd

    val validPredictionsAndLabelsCV = cvModel.transform(Preproessing.validData).select("label", "prediction")
      .map { case Row(label: Double, prediction: Double) => (label, prediction) }.rdd

    val trainRegressionMetricsCV = new RegressionMetrics(trainPredictionsAndLabels)
    val validRegressionMetricsCV = new RegressionMetrics(validPredictionsAndLabels)
    
    val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel]

    val resultsCV = "\n=====================================================================\n" +
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
      "=====================================================================\n" + 
      s"CV params explained: ${cvModel.explainParams}\n" +
      s"GLR params explained: ${bestModel.stages.last.asInstanceOf[GeneralizedLinearRegressionModel].explainParams}\n" +
      "=====================================================================\n"
    println(resultsCV)

    spark.stop()
  }
}