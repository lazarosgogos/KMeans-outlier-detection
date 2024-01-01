
import org.apache.spark.SparkContext._
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql._
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
object Main {
  def main(args: Array[String]): Unit = {
    /*val conf = new SparkConf()
      .setMaster("local[1]") // run using 2 threads, use local[*] to run with as many threads as possible
      .setAppName("DataMinerApp")*/
//    val sc = new SparkContext(conf)
    val ss = SparkSession.builder()
      .master("local[8]")
      .appName("DataMinerApp")
      .getOrCreate() // It's either SparkSession or SparkContext, choose your enemy!!
    val sc = ss.sparkContext

    val currentDir = System.getProperty("user.dir")
    //    val inputfile = "file://" + currentDir +"/points.csv"
    val inputfile = "points.csv"

    val txtFile = sc.textFile(inputfile).cache()

    val result = txtFile.map(line => line.split('\n').mkString) // split each line
      .filter(x => !x.startsWith(",") && !x.endsWith(",") && x.nonEmpty) // keep only valid points
      .map(line => {
        val splitted = line.split(",")
        (splitted(0).trim.toDouble, splitted(1).trim.toDouble) // convert to numbers
      }) // create tuples
      .collect() // create the RDD

    // find min/max values per column
    val minX = result.minBy(_._1)._1
    val minY = result.minBy(_._2)._2
    val maxX = result.maxBy(_._1)._1
    val maxY = result.maxBy(_._2)._2
//    println(minX)

    val data = result.map(d => {
      var x = d._1
      var y = d._2
      ((x - minX)/(maxX - minX), (y - minY)/(maxY - minY))
    }) // scale values

    // ----------------------- K MEANS -----------------------
    // for more info refer to:
    // https://spark.apache.org/docs/latest/ml-clustering.html
    val schema = StructType(
    StructField("x", DoubleType, false) ::
    StructField("y", DoubleType, false) :: Nil
    )
    // How and why?
    // here's the answer
    // https://stackoverflow.com/questions/41042809/tuple-to-data-frame-in-spark-scala
    val rdd = sc.parallelize(data)
      .map(line => Row(
        line._1.asInstanceOf[Number].doubleValue(),
        line._2.asInstanceOf[Number].doubleValue())
      )
    val df = ss.createDataFrame(rdd, schema)

    //    df.head(50).foreach(println)
    val kmeans = new KMeans().setK(20).setSeed(199L)

    //creating features column
    val assembler = new VectorAssembler()
      .setInputCols(Array("x", "y"))
      .setOutputCol("features")

    val features = assembler.transform(df)
    // Train the model
    val model = kmeans.fit(features)


    // Make predictions
    val predictions = model.transform(features)

    val evaluator = new ClusteringEvaluator()

    val silhouette = evaluator.evaluate(predictions)
    println(s"Silhouette with squared euclidean distance = $silhouette")

    // show the result
    println("Cluster Centers: ")
    model.clusterCenters.foreach(println)

  }
}