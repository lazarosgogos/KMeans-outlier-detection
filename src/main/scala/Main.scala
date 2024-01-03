
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
object Main {

  def main(args: Array[String]): Unit = {

    //    0. Initialize SparkSession and SparkContext
    val ss = initializeSparkSession(no_cores = 2, appName = "PapaGo_Clustering")
    val sc = ss.sparkContext

    //    1. Read input file
    val rawDataRDD = readInputDataFile(sc, args)

    //    2. Perform data cleaning
    val validCoordinatesRDD = performDataCleaning(rawDataRDD)

    //    3. Scale data to [0, 1]
    val coordinatesLimits = getCoordinatesLimits(validCoordinatesRDD) // List(xMin, xMax, yMin, yMax)
    val transformedCoordinatesRDD = performMinMaxScaling(validCoordinatesRDD, coordinatesLimits)

    // ----------------------- K MEANS -----------------------
    // for more info refer to:
    // https://spark.apache.org/docs/latest/ml-clustering.html

    val dataFrame = createDataFrameFromCoordinatesRDD(transformedCoordinatesRDD, ss)


    //    df.head(50).foreach(println)
    val kMeans = new KMeans().setK(20).setSeed(199L)

    //creating features column
    val assembler = new VectorAssembler()
      .setInputCols(Array("x", "y"))
      .setOutputCol("features")

    val features = assembler.transform(dataFrame)
    // Train the model
    val model = kMeans.fit(features)


    // Make predictions
    val predictions = model.transform(features)

    val evaluator = new ClusteringEvaluator()

    val silhouette = evaluator.evaluate(predictions)
    println(s"Silhouette with squared euclidean distance = $silhouette")

    // show the result
    println("Cluster Centers: ")
    model.clusterCenters.foreach(println)
    sc.stop(0)
  }

  private def initializeSparkSession(no_cores : Integer, appName : String) : SparkSession = {
    SparkSession.builder()
      .master(s"local[$no_cores]")
      .appName(appName)
      .getOrCreate()
  }

  private def getInputFileName(args: Array[String]) : String = {
    val defaultFileName = "points.csv"
    if (args.length == 0) {
      return defaultFileName
    }
    args(0)
  }


  //    ASSIGNMENT TASK 1
  private def readInputDataFile(sc : SparkContext, args:Array[String]) : RDD[String] = {
    val inputFile = getInputFileName(args)
    sc.textFile(inputFile)
  }

  private def filterOutLinesWithMissingValues(rawData : RDD[String]) : RDD[String] = {
    rawData.filter(line =>
      !line.startsWith(",") &&
      !line.endsWith(",") &&
      line.nonEmpty
    )
  }

  private def splitLinesToDoubleCoordinates(stringData : RDD[String]) : RDD[(Double, Double)] = {
    stringData.map(line => {
      val coordinates = line.split(",")
      (coordinates(0).toDouble, coordinates(1).toDouble)
    })
  }


  //    ASSIGNMENT TASK 2
  private def performDataCleaning(rawData : RDD[String]) : RDD[(Double, Double)] = {
    val cleanData = filterOutLinesWithMissingValues(rawData)
    splitLinesToDoubleCoordinates(cleanData)
  }


  //    ASSIGNMENT TASK 3
  private def performMinMaxScaling(unscaled : RDD[(Double, Double)], coordinatesLimits : List[Double]) : RDD[(Double, Double)] = {
    val xMin = coordinatesLimits.head
    val xMax = coordinatesLimits(1)
    val yMin = coordinatesLimits(2)
    val yMax = coordinatesLimits(3)
    unscaled.map(coordinates => {
      val x = coordinates._1
      val y = coordinates._2
      ((x - xMin) / (xMax - xMin), (y - yMin) / (yMax - yMin))
    })
  }

  private def getCoordinatesLimits(coordinates: RDD[(Double, Double)]): List[Double] = {
    val xCoordinatesRDD = coordinates.map(coordinates => coordinates._1)
    val yCoordinatesRDD = coordinates.map(coordinates => coordinates._2)
    val xMin = xCoordinatesRDD.min
    val xMax = xCoordinatesRDD.max
    val yMin = yCoordinatesRDD.min
    val yMax = yCoordinatesRDD.max

    List(xMin, xMax, yMin, yMax)
  }

  //    ASSIGNMENT TASK 4
  private def createDataFrameFromCoordinatesRDD(coordinates : RDD[(Double, Double)], ss : SparkSession) : DataFrame = {
    val schema = StructType(
      StructField(name = "x", dataType = DoubleType, nullable = false) ::
      StructField(name = "y", dataType = DoubleType, nullable = false) :: Nil
    )
    // How and why?
    // here's the answer
    // https://stackoverflow.com/questions/41042809/tuple-to-data-frame-in-spark-scala

    val data = coordinates.map(line => Row(
      line._1.asInstanceOf[Number].doubleValue(),
      line._2.asInstanceOf[Number].doubleValue()
    ))
    ss.createDataFrame(data, schema)
  }

}