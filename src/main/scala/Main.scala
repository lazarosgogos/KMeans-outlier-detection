import org.apache.spark.SparkContext
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}

import java.io.PrintWriter

object Main {
  private var xMin: Double = 0.0
  private var xMax: Double = 1.0
  private var yMin: Double = 0.0
  private var yMax: Double = 1.0

  private final val NANOS_PER_SEC: Double = 1e9d

  def main(args: Array[String]): Unit = {

    //    0. Initialize timer, SparkSession and SparkContext
    val startTimestamp = System.nanoTime
    val ss = initializeSparkSession(cores = "*", appName = "PapaGo_Clustering")
    val sc = ss.sparkContext
    sc.setLogLevel("ERROR")

    //    1. Read input file
    val rawData = readInputDataFile(sc, args)

    //    2. Perform data cleaning
    val validCoordinates = performDataCleaning(rawData)

    //    3. Scale data to [0, 1]
    computeMinMaxValues(validCoordinates) // computes xMin, xMax, yMin, yMax and saves them to object attributes
    val transformedCoordinates = performMinMaxScaling(validCoordinates)

    //   4. Implement K-means clustering for k >> 5   +   5. Detect Outliers
    val dataFrame = createDataFrameFromCoordinatesRDD(transformedCoordinates, ss)
    runKMeans(k = 200, seed = System.nanoTime(), dataFrame = dataFrame,
      predictionsDir = "predictions.d", centersOutFile = "kMeans_centers.csv",
      detectOutliers = true, removeOutliers = true
    )

    val duration = (System.nanoTime - startTimestamp) / NANOS_PER_SEC
    println(s"\nTotal execution time: $duration sec")
    sc.stop(0)
  }

  private def initializeSparkSession(cores: String, appName: String): SparkSession = {
    SparkSession.builder()
      .master(s"local[$cores]")
      .appName(appName)
      .getOrCreate()
  }

  private def getInputFileName(args: Array[String]): String = {
    val defaultFileName = "data.csv"
    if (args.length == 0) {
      return defaultFileName
    }
    args(0)
  }


  //    ASSIGNMENT TASK 1
  private def readInputDataFile(sc: SparkContext, args: Array[String]): RDD[String] = {
    val inputFile = getInputFileName(args)
    sc.textFile(inputFile)
  }

  private def filterOutLinesWithMissingValues(rawData: RDD[String]): RDD[String] = {
    rawData.filter(line =>
      !line.startsWith(",") &&
        !line.endsWith(",") &&
        line.nonEmpty
    )
  }

  //    ASSIGNMENT TASK 2
  private def performDataCleaning(rawData: RDD[String]): RDD[(Double, Double)] = {
    val cleanData = filterOutLinesWithMissingValues(rawData)
    splitLinesToDoubleCoordinates(cleanData)
  }

  private def splitLinesToDoubleCoordinates(stringData: RDD[String]): RDD[(Double, Double)] = {
    stringData.map(line => {
      val coordinates = line.split(",")
      (coordinates(0).toDouble, coordinates(1).toDouble)
    })
  }


  //    ASSIGNMENT TASK 3
  private def performMinMaxScaling(unscaled: RDD[(Double, Double)]): RDD[(Double, Double)] = {
    unscaled.map(coordinates => {
      val x = coordinates._1
      val y = coordinates._2
      ((x - xMin) / (xMax - xMin), (y - yMin) / (yMax - yMin))
    })
  }

  private def computeMinMaxValues(coordinates: RDD[(Double, Double)]): Unit = {
    val x = coordinates.map(coordinates => coordinates._1)
    val y = coordinates.map(coordinates => coordinates._2)
    xMin = x.min
    xMax = x.max
    yMin = y.min
    yMax = y.max
  }

  private def getOriginalCoordinates(xScaled: Double, yScaled: Double): Vector[Double] = {
    Vector(
      xScaled * (xMax - xMin) + xMin,
      yScaled * (yMax - yMin) + yMin
    )
  }


  //    ASSIGNMENT TASK 4
  private def createDataFrameFromCoordinatesRDD(coordinates: RDD[(Double, Double)], ss: SparkSession): DataFrame = {
    val schema = StructType(
      StructField(name = "x", dataType = DoubleType, nullable = false) ::
        StructField(name = "y", dataType = DoubleType, nullable = false) :: Nil
    )
    val data = coordinates.map(line => Row(
      line._1.asInstanceOf[Number].doubleValue(),
      line._2.asInstanceOf[Number].doubleValue()
    ))
    ss.createDataFrame(data, schema)
  }

  private def runKMeans(k: Integer = 10, seed: Long = 22L, dataFrame: DataFrame,
                        predictionsDir: String, centersOutFile: String,
                        detectOutliers: Boolean, removeOutliers: Boolean): Unit = {

    // Create features column
    val assembler = new VectorAssembler()
      .setInputCols(Array("x", "y"))
      .setOutputCol("features")
    val features = assembler.transform(dataFrame)

    // Train the model
    val kMeans = new KMeans().setK(k).setSeed(seed)
    val model = kMeans.fit(features)

    // Make predictions
    val predictions = model.transform(features).selectExpr("x as xScaled", "y as yScaled", "prediction as clusterId")
    writePredictions(dataFrame = predictions, outDir = predictionsDir)

    // Retrieve original coordinates of the cluster centers
    val originalCenters = model.clusterCenters.map(scaledCoordinates => getOriginalCoordinates(scaledCoordinates(0), scaledCoordinates(1)))
    writeCenters(centers = originalCenters, outFile = centersOutFile)

    val predictionsExpanded = predictions
      .withColumn("xOriginal", predictions("xScaled") * (xMax - xMin) + xMin)
      .withColumn("yOriginal", predictions("yScaled") * (yMax - yMin) + yMin)
      .rdd.map(row => {
        val xScaled = row.getDouble(0)
        val yScaled = row.getDouble(1)
        val clusterId = row.getInt(2)
        val xOriginal = row.getDouble(3)
        val yOriginal = row.getDouble(4)
        val xClusterCenter = originalCenters(clusterId)(0)
        val yClusterCenter = originalCenters(clusterId)(1)
        val euclideanDistanceFromCenter = math.sqrt(
          math.pow(xOriginal - xClusterCenter, 2) +
            math.pow(yOriginal - yClusterCenter, 2)
        )
        Row(xScaled, yScaled, clusterId, xOriginal, yOriginal,
          xClusterCenter, yClusterCenter, euclideanDistanceFromCenter)
      })

    val meanDistances = predictionsExpanded.map(row => (row.getInt(2), row.getDouble(7)))
      .mapValues(distance => (distance, 1))
      .reduceByKey((pair1, pair2) => (pair1._1 + pair2._1, pair1._2 + pair2._2))
      .mapValues(pair => pair._1 / pair._2)
      .collect()
      .toList
      .sortBy(pair => pair._1)

    val stdDistances = predictionsExpanded.map(row => (row.getInt(2), row.getDouble(7)))
      .mapValues(x => (1, x, x * x))
      .reduceByKey((x, y) => (x._1 + y._1, x._2 + y._2, x._3 + y._3))
      .mapValues(x => math.sqrt(x._3 / x._1 - math.pow(x._2 / x._1, 2)))
      .collect()
      .toList
      .sortBy(pair => pair._1)

    printf("Detected Outliers:\n\n")
    predictionsExpanded.filter(row => row.getDouble(7) > meanDistances(row.getInt(2))._2 + 3.5 * stdDistances(row.getInt(2))._2)
      .foreach(row => printf("(%.5f, %.5f)\n", row.getDouble(3), row.getDouble(4)))
  }

  private def writePredictions(dataFrame: DataFrame, outDir: String): Unit = {
    dataFrame.write.mode(SaveMode.Overwrite).csv(outDir)
  }

  private def writeCenters(centers: Array[Vector[Double]], outFile: String): Unit = {
    writeToFile(outFile) { printer =>
      printer.println("x,y,clusterId")
      centers.zipWithIndex.foreach(vector => printer.printf("%f,%f,%d\n", vector._1(0), vector._1(1), vector._2))
    }
  }

  private def writeToFile(outFile: String)(op: java.io.PrintWriter => Unit): Unit = {
    val p = new PrintWriter(outFile)
    try {
      op(p)
    } finally {
      p.close()
    }
  }
}