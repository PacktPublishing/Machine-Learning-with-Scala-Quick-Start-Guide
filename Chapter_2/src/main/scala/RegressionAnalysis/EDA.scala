package RegressionAnalysis

import org.apache.spark.sql._
import org.apache.spark.sql.functions._

object EDA {
  def main(args: Array[String]): Unit = {
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

    rawTrafficDF.select("Hour (Coded)", "Immobilized bus", "Broken Truck", "Vehicle excess", "Fire", "Slowness in traffic (%)").show(5)
    println(rawTrafficDF.count())
    rawTrafficDF.printSchema()
    
    rawTrafficDF.select("Hour (Coded)", "Immobilized bus", "Broken Truck", "Point of flooding", "Fire", "Slowness in traffic (%)").describe().show()

    var newTrafficDF = rawTrafficDF.withColumnRenamed("Slowness in traffic (%)", "label")

    // Let's explore two other important features Point of flooding and Vehicle excess. We can rename these two columns as follows: 

    newTrafficDF = newTrafficDF.withColumnRenamed("Point of flooding", "NoOfFloodPoint")

    newTrafficDF.createOrReplaceTempView("slDF")
    spark.sql("SELECT avg(label) as avgSlowness FROM slDF").show()

    spark.sql("SELECT max(NoOfFloodPoint) FROM slDF").show()
  }
}