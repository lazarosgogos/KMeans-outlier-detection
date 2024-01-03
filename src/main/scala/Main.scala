import org.apache.spark.SparkContext
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}

import java.io.PrintWriter

object Main {
  private var xMin : Double = 0.0
  private var xMax : Double = 1.0
  private var yMin : Double = 0.0
  private var yMax : Double = 1.0

   def main(args: Array[String]): Unit = {

    //    0. Initialize timer, SparkSession and SparkContext
    val startTimestamp = System.nanoTime
    val ss = initializeSparkSession(cores = "*", appName = "PapaGo_Clustering")
    val sc = ss.sparkContext
    sc.setLogLevel("OFF")

    //    1. Read input file
    val rawData = readInputDataFile(sc, args)

    //    2. Perform data cleaning
    val validCoordinates = performDataCleaning(rawData)

    //    3. Scale data to [0, 1]
    computeMinMaxValues(validCoordinates) // computes xMin, xMax, yMin, yMax and saves them to object attributes
    val transformedCoordinates = performMinMaxScaling(validCoordinates)

     //   4. Implement K-means clustering
    val dataFrame = createDataFrameFromCoordinatesRDD(transformedCoordinates, ss)
    runKMeans(k = 50, seed = 1530L, dataFrame = dataFrame, predictionsDir = "predictions.d" , centersOutFile = "kMeans_centers.txt")
    val duration = (System.nanoTime - startTimestamp) / 1e9d
    println(s"Total execution time: $duration sec")
    sc.stop(0)
  }

  private def initializeSparkSession(cores : String, appName : String) : SparkSession = {
    SparkSession.builder()
      .master(s"local[$cores]")
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

  //    ASSIGNMENT TASK 2
  private def performDataCleaning(rawData : RDD[String]) : RDD[(Double, Double)] = {
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
  private def performMinMaxScaling(unscaled : RDD[(Double, Double)]) : RDD[(Double, Double)] = {
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

  private def getOriginalCoordinates(xScaled : Double, yScaled : Double) : Vector[Double] = {
    Vector(
      xScaled * (xMax - xMin) + xMin,
      yScaled * (yMax - yMin) + yMin
    )
  }


  //    ASSIGNMENT TASK 4
  private def createDataFrameFromCoordinatesRDD(coordinates : RDD[(Double, Double)], ss : SparkSession) : DataFrame = {
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

  private def runKMeans(k: Integer = 10, seed: Long = 199L, dataFrame: DataFrame,
                        predictionsDir : String , centersOutFile : String): Unit = {

    val kMeans = new KMeans().setK(k).setSeed(seed)

    //creating features column
    val assembler = new VectorAssembler()
      .setInputCols(Array("x", "y"))
      .setOutputCol("features")

    val features = assembler.transform(dataFrame)
    // Train the model
    val model = kMeans.fit(features)

    // Make predictions
    val predictions = model.transform(features)
    writePredictions(dataFrame = predictions, outDir = predictionsDir)

    // Retrieve Original Centers and write them to a txt file for future use
    val originalCenters = model.clusterCenters.map(scaledCoordinates => getOriginalCoordinates(scaledCoordinates(0), scaledCoordinates(1)))
    writeCenters(centers = originalCenters, outFile = centersOutFile)
  }

  private def writePredictions(dataFrame : DataFrame, outDir : String) : Unit = {
    dataFrame.select("x", "y", "prediction").select("x", "y", "prediction").write.mode(SaveMode.Overwrite).csv(outDir)
  }

  private def writeCenters(centers : Array[Vector[Double]], outFile : String) : Unit = {
    writeToFile(outFile) { printer =>
      centers.zipWithIndex.foreach(vector => printer.printf("centers.append([%f, %f, %d])\n", vector._1(0), vector._1(1), vector._2))
    }
  }

  def writeToFile(outFile: String)(op: java.io.PrintWriter => Unit) {
    val p = new PrintWriter(outFile)
    try { op(p) } finally { p.close() }
  }
}