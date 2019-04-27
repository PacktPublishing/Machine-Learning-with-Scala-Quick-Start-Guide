package ScalaTreeEnsimbles

import org.apache.spark.ml.feature.{ StringIndexer, StringIndexerModel }
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql._
import org.apache.spark.sql.functions._

object Preproessing {
  var trainSample = 1.0
  var testSample = 1.0
  val train = "data/insurance_train.csv"
  val test = "data/insurance_test.csv"

  val spark = SparkSession
    .builder
    .master("local[*]")
    .config("spark.sql.warehouse.dir", "E:/Exp/")
    .appName(s"OneVsRestExample")
    .getOrCreate()
  
  import spark.implicits._
  println("Reading data from " + train + " file")

  val trainInput = spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .format("com.databricks.spark.csv")
    .load(train)
    .cache

  val testInput = spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .format("com.databricks.spark.csv")
    .load(test)
    .cache

  println("Preparing data for training model")
  var data = trainInput.withColumnRenamed("loss", "label").sample(false, trainSample)
  var DF = data.na.drop()

  // Null check
  if (data == DF)
    println("No null values in the DataFrame")

  else {
    println("Null values exist in the DataFrame")
    data = DF
  }

  val seed = 23579L
  val splits = data.randomSplit(Array(0.80, 0.20), seed)
  val (trainingData, validData) = (splits(0), splits(1))

  trainingData.cache
  validData.cache

  val testData = testInput.sample(false, testSample).cache

  def isCateg(c: String): Boolean = c.startsWith("cat")
  def categNewCol(c: String): String = if (isCateg(c)) s"idx_${c}" else c

  // Function to remove categorical columns with too many categories
  def removeTooManyCategs(c: String): Boolean = !(c matches "cat(109$|110$|112$|113$|116$)")

  // Function to select only feature columns (omit id and label)
  def onlyFeatureCols(c: String): Boolean = !(c matches "id|label")

  // Definitive set of feature columns
  val featureCols = trainingData.columns
    .filter(removeTooManyCategs)
    .filter(onlyFeatureCols)
    .map(categNewCol)

  // StringIndexer for categorical columns (OneHotEncoder should be evaluated as well)
  val stringIndexerStages = trainingData.columns.filter(isCateg)
    .map(c => new StringIndexer()
      .setInputCol(c)
      .setOutputCol(categNewCol(c))
      .fit(trainInput.select(c).union(testInput.select(c))))

  // VectorAssembler for training features
  val assembler = new VectorAssembler()
    .setInputCols(featureCols)
    .setOutputCol("features")
}