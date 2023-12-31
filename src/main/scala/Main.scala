
import org.apache.spark.SparkContext._
import org.apache.spark.{SparkConf, SparkContext}
//import org.apache.spark.sql.SparkSession

object Main {
  def main(args: Array[String]): Unit = {

    val conf = new SparkConf()
      .setMaster("local[1]") // run using 2 threads, use local[*] to run with as many threads as possible
      .setAppName("DataMinerApp")

    val sc = new SparkContext(conf)
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
//      x = ((x-minX) / (maxX - minX))

      ((x - minX)/(maxX - minX), (y - minY)/(maxY - minY))
    })


  }
}