package ScalaTreeEnsimbles

import org.apache.spark.ml.regression.{ GBTRegressor, GBTRegressionModel }
import org.apache.spark.ml.{ Pipeline, PipelineModel }
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.log4j.LogManager
import org.apache.spark.ml.tuning.{ CrossValidator, ParamGridBuilder }
import org.apache.spark.ml.feature.VectorAssembler


object UrbanTrafficGBTRegressor {
  def main(args: Array[String]) {
    val spark = SparkSession
      .builder
      .master("local[*]")
      .config("spark.sql.warehouse.dir", "E:/Exp/")
      .appName(s"OneVsRestExample")
      .getOrCreate()

    import spark.implicits._

    val rawTrafficDF = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .option("delimiter", ";")
      .format("com.databricks.spark.csv")
      .load("data/Behavior of the urban traffic of the city of Sao Paulo in Brazil.csv")
      .cache
    
    val newTrafficDF = rawTrafficDF.withColumnRenamed("Slowness in traffic (%)", "label")     
    val colNames = newTrafficDF.columns.dropRight(1)

    // VectorAssembler for training features
    val assembler = new VectorAssembler()
      .setInputCols(colNames)
      .setOutputCol("features")

    val assembleDF = assembler.transform(newTrafficDF).select("features", "label")  
    assembleDF.printSchema()
    
    val seed = 12345
    val splits = assembleDF.randomSplit(Array(0.60, 0.40), seed)
    val (trainingData, testData) = (splits(0), splits(1))

    trainingData.cache
    testData.cache    
    
    // Estimator algorithm
    val gbtModel = new GBTRegressor().setFeaturesCol("features").setLabelCol("label")

    // ***********************************************************
    println("Preparing K-fold Cross Validation and Grid Search")
    // ***********************************************************

    // Search through decision tree's maxDepth parameter for best model
    var paramGrid = new ParamGridBuilder()
      .addGrid(gbtModel.impurity, "variance" :: Nil)// variance for regression
      .addGrid(gbtModel.maxBins, 3 :: 5 :: 10 :: Nil)
      .addGrid(gbtModel.maxDepth, 2 :: 5 :: 10 :: Nil)
      .build()

    val numFolds = 10  
    val cv = new CrossValidator()
      .setEstimator(gbtModel)
      .setEvaluator(new RegressionEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(numFolds)

    // ************************************************************
    println("Training model with GradientBoostedTrees algorithm")
    // ************************************************************
    val cvModel = cv.fit(trainingData)

    // **********************************************************************
    println("Evaluating the model on the test set and calculating the regression metrics")
    // **********************************************************************
    val trainPredictionsAndLabels = cvModel.transform(testData).select("label", "prediction")
                                            .map { case Row(label: Double, prediction: Double) => (label, prediction) }.rdd

    val testRegressionMetrics = new RegressionMetrics(trainPredictionsAndLabels)

    val results = "\n=====================================================================\n" +
      s"TrainingData count: ${trainingData.count}\n" +
      s"TestData count: ${testData.count}\n" +
      "=====================================================================\n" +
      s"TestData MSE = ${testRegressionMetrics.meanSquaredError}\n" +
      s"TestData RMSE = ${testRegressionMetrics.rootMeanSquaredError}\n" +
      s"TestData R-squared = ${testRegressionMetrics.r2}\n" +
      s"TestData MAE = ${testRegressionMetrics.meanAbsoluteError}\n" +
      s"TestData explained variance = ${testRegressionMetrics.explainedVariance}\n" +
      "=====================================================================\n"
    println(results)

    val bestModel = cvModel.bestModel.asInstanceOf[GBTRegressionModel]
      
    println("Decison tree from best cross-validated model" + bestModel.toDebugString)

    val featureImportances = bestModel.featureImportances.toArray

    val FI_to_List_sorted = featureImportances.toList.sorted.toArray
    println("Feature importance generated by the best model: ")
    for(x <- FI_to_List_sorted) println(x)
  }
}