package ScalaBookRecommendation

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.SQLImplicits
import org.apache.spark.sql._
import org.apache.spark.sql.Dataset
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating
import scala.Tuple2

import org.apache.spark.rdd.RDD

object BookRecommendation {
  //Compute the RMSE to evaluate the model. Less the RMSE better the model and it's prediction capability. 
  def computeRmse(model: MatrixFactorizationModel, data: RDD[Rating], implicitPrefs: Boolean): Double = {
    val predictions: RDD[Rating] = model.predict(data.map(x => (x.user, x.product)))
    val predictionsAndRatings = predictions.map { x => ((x.user, x.product), x.rating)
    }.join(data.map(x => ((x.user, x.product), x.rating))).values
    math.sqrt(predictionsAndRatings.map(x => (x._1 - x._2) * (x._1 - x._2)).mean())
  }

  def main(args: Array[String]) {
    val spark = SparkSession
      .builder
      .master("local[*]")
      .config("spark.sql.warehouse.dir", "E:/Exp/")
      .appName("BookRecommendation")
      .getOrCreate()

    import spark.implicits._

    println("Loading Ratings data...")

    val ratigsFile = "data/BX-Book-Ratings.csv"
    var ratingDF = spark.read.format("com.databricks.spark.csv")
      .option("delimiter", ";")
      .option("header", true)
      .load(ratigsFile)

    ratingDF = ratingDF.withColumnRenamed("User-ID", "UserID").withColumnRenamed("Book-Rating", "Rating")
    ratingDF.printSchema()

    /* Explore and Query with Spark DataFrames		 */
    val numRatings = ratingDF.count()
    val numUsers = ratingDF.select(ratingDF.col("UserID")).distinct().count()
    val numBooks = ratingDF.select(ratingDF.col("ISBN")).distinct().count()
    println("Got " + numRatings + " ratings from " + numUsers + " users on " + numBooks + " books") /* Got 1149780 ratings from 105283 users on 340556 books */

    val booksFile = "data/BX-Books.csv"
    var bookDF = spark.read.format("com.databricks.spark.csv").option("header", "true").option("delimiter", ";").load(booksFile)
    bookDF.show()

    bookDF = bookDF.select(bookDF.col("ISBN"), bookDF.col("Book-Title"), bookDF.col("Book-Author"), bookDF.col("Year-Of-Publication"))
    bookDF = bookDF.withColumnRenamed("Book-Title", "Title").withColumnRenamed("Book-Author", "Author").withColumnRenamed("Year-Of-Publication", "Year")
    bookDF.show(10)
    /*
     * +----------+--------------------+--------------------+----+
        |      ISBN|               Title|              Author|Year|
        +----------+--------------------+--------------------+----+
        |0195153448| Classical Mythology|  Mark P. O. Morford|2002|
        |0002005018|        Clara Callan|Richard Bruce Wright|2001|
        |0060973129|Decision in Normandy|        Carlo D'Este|1991|
        |0374157065|Flu: The Story of...|    Gina Bari Kolata|1999|
        |0393045218|The Mummies of Ur...|     E. J. W. Barber|1999|
        |0399135782|The Kitchen God's...|             Amy Tan|1991|
        |0425176428|What If?: The Wor...|       Robert Cowley|2000|
        |0671870432|     PLEADING GUILTY|         Scott Turow|1993|
        |0679425608|Under the Black F...|     David Cordingly|1996|
        |074322678X|Where You'll Find...|         Ann Beattie|2002|
        +----------+--------------------+--------------------+----+
        only showing top 10 rows
     */

    ratingDF.createOrReplaceTempView("ratings")
    bookDF.createOrReplaceTempView("books")
    
    spark.sql("SELECT max(Rating) FROM ratings").show()

    // Get the max, min ratings along with the count of users who have rated a book.
    val statDF = spark.sql("select books.Title, bookrates.maxr, bookrates.minr, bookrates.cntu "
      + "from(SELECT ratings.ISBN,max(ratings.Rating) as maxr,"
      + "min(ratings.Rating) as minr,count(distinct UserID) as cntu "
      + "FROM ratings group by ratings.ISBN) bookrates "
      + "join books on bookrates.ISBN=books.ISBN " + "order by bookrates.cntu desc")

    statDF.show(10)
    /*
     * +--------------------+----+----+----+
        |               Title|maxr|minr|cntu|
        +--------------------+----+----+----+
        |         Wild Animus|   9|   0|2502|
        |The Lovely Bones:...|   9|   0|1295|
        |   The Da Vinci Code|   9|   0| 883|
        |Divine Secrets of...|   9|   0| 732|
        |The Red Tent (Bes...|   9|   0| 723|
        |     A Painted House|   9|   0| 647|
        |The Secret Life o...|   9|   0| 615|
        |Snow Falling on C...|   9|   0| 614|
        | Angels &amp; Demons|   9|   0| 586|
        |Where the Heart I...|   9|   0| 585|
        +--------------------+----+----+----+
        only showing top 10 rows
     */

    // Show the top 10 most-active users and how many times they rated a book
    val mostActiveReaders = spark.sql("SELECT ratings.UserID, count(*) as CT from ratings "
      + "group by ratings.UserID order by CT desc limit 10")
    mostActiveReaders.show()
    /*
     * +------+-----+
      |UserID|   CT|
      +------+-----+
      | 11676|13602|
      |198711| 7550|
      |153662| 6109|
      | 98391| 5891|
      | 35859| 5850|
      |212898| 4785|
      |278418| 4533|
      | 76352| 3367|
      |110973| 3100|
      |235105| 3067|
      +------+-----+
     */

    // Find the movies that user 276744 rated higher than 5
    val ratingBySpecificReader = spark.sql(
      "SELECT ratings.UserID, ratings.ISBN,"
        + "ratings.Rating, books.Title FROM ratings JOIN books "
        + "ON books.ISBN=ratings.ISBN "
        + "where ratings.UserID=276744 and ratings.Rating > 4")

    ratingBySpecificReader.show(false)

    /*
     * +------+----------+------+---------------+
        |UserID|ISBN      |Rating|Title          |
        +------+----------+------+---------------+
        |276744|038550120X|7     |A Painted House|
        +------+----------+------+---------------+
     */

    // Feature engineering     
    ratingDF = ratingDF.withColumn("ISBN_1", hash($"ISBN"))
    ratingDF = ratingDF.select("UserID", "ISBN_1", "Rating")
    ratingDF = ratingDF.withColumn("ISBN", abs($"ISBN_1"))
    ratingDF = ratingDF.select("UserID", "ISBN", "Rating")

    ratingDF.printSchema()
    /*
     * root
         |-- UserID: string (nullable = true)
         |-- ISBN: integer (nullable = false)
         |-- Rating: string (nullable = true)
     */

    val seed = 12345
    val splits = ratingDF.randomSplit(Array(0.60, 0.40), seed)
    val (trainingData, testData) = (splits(0), splits(1))

    trainingData.cache
    testData.cache

    val numTrainingSample = trainingData.count()
    val numTestSample = testData.count()
    println("Training: " + numTrainingSample + " test: " + numTestSample) // Training: 689144 test: 345774    

    val trainRatingsRDD = trainingData.rdd.map(row => {
      val userID = row.getString(0)
      val ISBN = row.getInt(1)
      val ratings = row.getString(2)
      Rating(userID.toInt, ISBN, ratings.toDouble)
    })

    val testRatingsRDD = testData.rdd.map(row => {
      val userID = row.getString(0)
      val ISBN = row.getInt(1)
      val ratings = row.getString(2)
      Rating(userID.toInt, ISBN, ratings.toDouble)
    })

    val model : MatrixFactorizationModel = new ALS()
      .setIterations(10)
      .setBlocks(-1)
      .setAlpha(1.0)
      .setLambda(0.01)
      .setRank(25)
      .setSeed(1234579L)
      .setImplicitPrefs(false)
      .run(trainRatingsRDD)

    //Saving the model for future use
    //val savedALSModel = model.save(spark.sparkContext, "model/MovieRecomModel")
      
    //Load the workflow back
    //val same_model = MatrixFactorizationModel.load(spark.sparkContext, "model/MovieRecomModel/")

    //Book recommendation for a specific user. Get the top 10 book predictions for reader 276747
    println("Recommendations: (ISBN, Rating)")
    println("----------------------------------")
    val recommendationsUser = model.recommendProducts(276747, 10)
    recommendationsUser.map(rating => (rating.product, rating.rating)).foreach(println)
    println("----------------------------------")

    /*
    Recommendations: (ISBN => Rating)
    (1051401851,15.127044702142243)
    (2056910662,15.11531283195148)
    (1013412890,14.75898119158678)
    (603241602,14.53024153450836)
    (1868529062,14.180262929540024)
    (746990712,14.121654522195225)
    (1630827789,13.741728003481194)
    (1179316963,13.571754513473993)
    (505970947,13.506755847456258)
    (632523982,13.46591014905454)
    ----------------------------------
     */

    // Evaluating the Model: we expect lower RMSE because smaller the calculated error, the better the model
    var rmseTest = computeRmse(model, testRatingsRDD, true)
    println("Test RMSE: = " + rmseTest) //Less is better // Test RMSE: = 1.6867585251053991   

    val new_user_ID = 300000 // new user ID randomly chosen

    //The format of each line is (UserID, ISBN, Rating)
    val new_user_ratings = Seq(
      (new_user_ID, 817930596, 15.127044702142243),
      (new_user_ID, 1149373895, 15.11531283195148),
      (new_user_ID, 1885291767, 14.75898119158678),
      (new_user_ID, 459716613, 14.53024153450836),
      (new_user_ID, 3362860, 14.180262929540024),
      (new_user_ID, 1178102612, 14.121654522195225),
      (new_user_ID, 158895996, 13.741728003481194),
      (new_user_ID, 1007741925, 13.571754513473993),
      (new_user_ID, 1033268461, 13.506755847456258),
      (new_user_ID, 651677816, 13.46591014905454))

    val new_user_ratings_RDD = spark.sparkContext.parallelize(new_user_ratings)
    val new_user_ratings_DF = spark.createDataFrame(new_user_ratings_RDD).toDF("UserID", "ISBN", "Rating")

    val newRatingsRDD = new_user_ratings_DF.rdd.map(row => {
      val userId = row.getInt(0)
      val movieId = row.getInt(1)
      val ratings = row.getDouble(2)
      Rating(userId, movieId, ratings)
    })

    val complete_data_with_new_ratings_RDD = trainRatingsRDD.union(newRatingsRDD)

    val newModel : MatrixFactorizationModel = new ALS()
      .setIterations(10)
      .setBlocks(-1)
      .setAlpha(1.0)
      .setLambda(0.01)
      .setRank(25)
      .setSeed(123457L)
      .setImplicitPrefs(false)
      .run(complete_data_with_new_ratings_RDD)

    // Making Predictions. Get the top 10 book predictions for user 276724
    //Book recommendation for a specific user. Get the top 10 book predictions for reader 276747
    println("Recommendations: (ISBN, Rating)")
    println("----------------------------------")
    val newPredictions = newModel.recommendProducts(276747, 10)
    newPredictions.map(rating => (rating.product, rating.rating)).foreach(println)
    println("----------------------------------")

    var newrmseTest = computeRmse(newModel, testRatingsRDD, true)
    println("Test RMSE: = " + newrmseTest) //Less is better
  }
}