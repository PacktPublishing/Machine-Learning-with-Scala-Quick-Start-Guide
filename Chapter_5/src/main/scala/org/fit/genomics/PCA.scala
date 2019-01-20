package org.fit.genomics

import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.PCA
import org.apache.spark.ml.linalg.Vectors

object PCAExample {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder.appName("PopStrat").master("local[*]").getOrCreate()

    val data = Array(
      Vectors.dense(3.5, 2.0, 5.0, 6.3, 5.60, 2.4),
      Vectors.dense(4.40, 0.10, 3.0, 9.0, 7.0, 8.75),
      Vectors.dense(3.20, 2.40, 0.0, 6.0, 7.4, 3.34))

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")
    df.show(false)

    val pca = new PCA()
      .setInputCol("features")
      .setOutputCol("pcaFeatures")
      .setK(4)
      .fit(df)

    val result = pca.transform(df).select("pcaFeatures")
    result.show(false)

  }
}