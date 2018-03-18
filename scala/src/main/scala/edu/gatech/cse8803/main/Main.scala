/**
 * @author Hang Su <hangsu@gatech.edu>.
 */

package edu.gatech.cse8803.main

import java.text.{NumberFormat, SimpleDateFormat}

import edu.gatech.cse8803.clustering.{Metrics, NMF}
import edu.gatech.cse8803.features.FeatureConstruction
import edu.gatech.cse8803.ioutils.CSVUtils
import edu.gatech.cse8803.model._
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.clustering.{GaussianMixture, KMeans, StreamingKMeans}
import org.apache.spark.mllib.linalg.{DenseMatrix, Matrices, Vector, Vectors}
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

import scala.io.Source


object Main {
  def main(args: Array[String]) {
    import org.apache.log4j.Logger
    import org.apache.log4j.Level

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val sc = createContext
    val sqlContext = new SQLContext(sc)

    /** initialize loading of data */
    val (socEcoRDD, scDemogrRDD, bioChemRDD, datScanRDD, upsitRDD, remRDD, moCARDD, updrsIRDD,updrsIIRDD, patientsWithLabel) = loadRddRawData(sqlContext)

    /** conduct phenotyping */
    val phenotypeLabel = patientsWithLabel



    /** feature construction with all features except UPDRS */
 /*   val featureTuples = sc.union(
      FeatureConstruction.constructSocEcoFeatureTuple(socEcoRDD),
      FeatureConstruction.constructScDemogrFeatureTuple(scDemogrRDD),
      FeatureConstruction.constructBioChemRDDFeatureTuple(bioChemRDD),
      FeatureConstruction.constructDatScanRDDFeatureTuple(datScanRDD),
      FeatureConstruction.constructUpsitRDDFeatureTuple(upsitRDD),
      FeatureConstruction.constructRemRDDFeatureTuple(remRDD),
      FeatureConstruction.constructMoCARDDFeatureTuple(moCARDD)
    )*/

    /** feature construction with all features */
    val featureTuples = sc.union(
      FeatureConstruction.constructSocEcoFeatureTuple(socEcoRDD),
      FeatureConstruction.constructScDemogrFeatureTuple(scDemogrRDD),
      FeatureConstruction.constructBioChemRDDFeatureTuple(bioChemRDD),
      FeatureConstruction.constructDatScanRDDFeatureTuple(datScanRDD),
      FeatureConstruction.constructUpsitRDDFeatureTuple(upsitRDD),
      FeatureConstruction.constructRemRDDFeatureTuple(remRDD),
      FeatureConstruction.constructMoCARDDFeatureTuple(moCARDD),
      FeatureConstruction.constructUpdrsIRDDFeatureTuple(updrsIRDD),
      FeatureConstruction.constructUpdrsIIRDDFeatureTuple(updrsIIRDD)
    )

    val rawFeatures = FeatureConstruction.construct(sc, featureTuples, phenotypeLabel)

    val (kMeansPurity, gaussianMixturePurity, streamKmeansPurity, nmfPurity) = testClustering(phenotypeLabel, rawFeatures)
    println(f"[All feature] purity of kMeans is: $kMeansPurity%.5f")
    println(f"[All feature] purity of GMM is: $gaussianMixturePurity%.5f")
    println(f"[All feature] purity of StreamingKMeans is: $streamKmeansPurity%.5f")
    println(f"[All feature] purity of NMF is: $nmfPurity%.5f")

    sc.stop 
  }

  def testClustering(phenotypeLabel: RDD[(String, Int)], rawFeatures:RDD[(String, Vector)]): (Double, Double, Double, Double) = {
    import org.apache.spark.mllib.linalg.Matrix
    import org.apache.spark.mllib.linalg.distributed.RowMatrix

    /** scale features */
    val scaler = new StandardScaler(withMean = true, withStd = true).fit(rawFeatures.map(_._2))
    val features = rawFeatures.map({ case (patientID, featureVector) => (patientID, scaler.transform(Vectors.dense(featureVector.toArray)))})
    val rawFeatureVectors = features.map(_._2).cache()

    /** reduce dimension */
    val mat: RowMatrix = new RowMatrix(rawFeatureVectors)
    val pc: Matrix = mat.computePrincipalComponents(10) // Principal components are stored in a local dense matrix.
    val featureVectors = mat.multiply(pc).rows

    val densePc = Matrices.dense(pc.numRows, pc.numCols, pc.toArray).asInstanceOf[DenseMatrix]
    /** transform a feature into its reduced dimension representation */
    def transform(feature: Vector): Vector = {
      Vectors.dense(Matrices.dense(1, feature.size, feature.toArray).multiply(densePc).toArray)
    }

    val k = 3 //cluster number

    /** TODO: K Means Clustering using spark mllib**/
    val KMmodel = KMeans.train(data = featureVectors, k=k, maxIterations = 20, runs = 1, initializationMode = "k-means||", seed = 8803L)
    val KMpred = KMmodel.predict(featureVectors)
    val KMpredWithPatientIds=features.map({case (patientId,f)=>patientId}).zip(KMpred)
    val KMpredictionAndLabel = KMpredWithPatientIds.join(phenotypeLabel).map({case (patientID,value)=>value})
    val kMeansPurity = Metrics.purity(KMpredictionAndLabel)


    /** TODO: GMMM Clustering using spark mllib**/
    val GMmodel = new GaussianMixture().setK(k).setMaxIterations(20).setSeed(8803L).run(featureVectors)
    val GMpred = GMmodel.predict(featureVectors)
    val GMpredWithPatientIds=features.map({case (patientId,f)=>patientId}).zip(GMpred)
    val GMpredictionAndLabel = GMpredWithPatientIds.join(phenotypeLabel).map({case (patientID,value)=>value})
    val gaussianMixturePurity = Metrics.purity(GMpredictionAndLabel)

    /** TODO: StreamingKMeans Clustering using spark mllib**/
    val SKMmodel = new StreamingKMeans().setK(k).setDecayFactor(1.0).setRandomCenters(weight = 0.0, seed = 8803L, dim = 10).
      latestModel().update(data = featureVectors, decayFactor = 1.0, timeUnit = "batches")
    val SKMpred = SKMmodel.predict(featureVectors)
    val SKMpredWithPatientIds=features.map({case (patientId,f)=>patientId}).zip(SKMpred)
    val SKMpredictionAndLabel = SKMpredWithPatientIds.join(phenotypeLabel).map({case (patientID,value)=>value})
    val streamKmeansPurity = Metrics.purity(SKMpredictionAndLabel)

    /** NMF */
/*    val rawFeaturesNonnegative = rawFeatures.map({ case (patientID, f)=> Vectors.dense(f.toArray.map(v=>Math.abs(v)))})
    val (w, _) = NMF.run(new RowMatrix(rawFeaturesNonnegative), 4, 100)
    // for each row (patient) in W matrix, the index with the max value should be assigned as its cluster type
    val assignments = w.rows.map(_.toArray.zipWithIndex.maxBy(_._1)._2)
    // zip patientIDs with their corresponding cluster assignments
    // Note that map doesn't change the order of rows
    val assignmentsWithPatientIds=features.map({case (patientId,f)=>patientId}).zip(assignments) 
    // join your cluster assignments and phenotypeLabel on the patientID and obtain a RDD[(Int,Int)]
    // which is a RDD of (clusterNumber, phenotypeLabel) pairs 
    val nmfClusterAssignmentAndLabel = assignmentsWithPatientIds.join(phenotypeLabel).map({case (patientID,value)=>value})
    // Obtain purity value
    val nmfPurity = Metrics.purity(nmfClusterAssignmentAndLabel)*/


/*    val kMeansPurity = 0.0
    val gaussianMixturePurity = 0.0
    val streamKmeansPurity = 0.0*/
    val nmfPurity = 0.0

    (kMeansPurity, gaussianMixturePurity, streamKmeansPurity, nmfPurity)
  }


  def loadRddRawData(sqlContext: SQLContext): (RDD[EDU], RDD[DEMOGR], RDD[BIOCHEM], RDD[DATSCAN], RDD[UPSIT], RDD[REM], RDD[MOCA], RDD[UPDRS], RDD[UPDRS], RDD[(String, Int)]) = {
    /** load data using Spark SQL into RDDs and return them
      *       Ignore lab results with missing (empty or NaN) values when these are read in.
      * */

    val patientStatus = CSVUtils.loadCSVAsTable(sqlContext,"data/Patient_Status.csv").
      toDF().select("PATNO","ENROLL_DATE","ENROLL_CAT","ENROLL_STATUS","DESCRP_CAT")
    val patientEnrolled = patientStatus.filter(patientStatus("ENROLL_STATUS")=== "Enrolled" || patientStatus("ENROLL_STATUS")=== "Withdrew").cache() // and withdraw

    val allPatients = patientEnrolled.filter(patientEnrolled("ENROLL_CAT")==="HC"||patientEnrolled("ENROLL_CAT")==="PD") //Healthy or Parkinson's Disease
    val allPatientsID = allPatients.map(s=>s(0)).distinct().collect()
    val patientsWithLabel = allPatients.rdd.map(s=>
      if (s(2) == "HC") (s(0).toString, 0)
      else (s(0).toString, 1) //PD
    )

    val socEco = CSVUtils.loadCSVAsTable(sqlContext,"data/Socio-Economics.csv").
      toDF().select("PATNO","EDUCYRS").cache() //Event ID & F_status

    val scDemogr = CSVUtils.loadCSVAsTable(sqlContext,"data/Screening___Demographics.csv").
      toDF().select("PATNO","APPRDX","BIRTHDT","GENDER","PRJENRDT","DECLINED","EXCLUDED") // "PRJENRDT" is "projected enrollment date"
    val scDemogrEnrolled = scDemogr.filter(!(scDemogr("EXCLUDED") === "1" || scDemogr("DECLINED") === "1")).cache()

    val bioChem = CSVUtils.loadCSVAsTable(sqlContext,"data/Biospecimen_Analysis_Results.csv").
      toDF().select("PATNO","GENDER","DIAGNOSIS","CLINICAL_EVENT","TYPE","TESTNAME","TESTVALUE").cache() //DNA RNA biochemical
    val bioChemFilterd = bioChem.filter((bioChem("TESTVALUE") !== "") and bioChem("TESTVALUE").isNotNull and (bioChem("TESTVALUE")!== "Undetermined") and (bioChem("TESTVALUE") !== "N/A") and (bioChem("TESTVALUE") !== "NA"))

    val datScan = CSVUtils.loadCSVAsTable(sqlContext,"data/DATScan_Analysis.csv").
      toDF().select("PATNO","EVENT_ID","CAUDATE_R","CAUDATE_L","PUTAMEN_R","PUTAMEN_L").cache().
      map(s=>(s(0).toString, s(1).toString, List(s(2), s(3)).map(_.toString).map(_.toDouble).sum, List(s(4), s(5)).map(_.toString).map(_.toDouble).sum))

    val upsit0 = CSVUtils.loadCSVAsTable(sqlContext,"data/University_of_Pennsylvania_Smell_ID_Test.csv").
      toDF().select("PATNO","EVENT_ID","UPSITBK1","UPSITBK2","UPSITBK3","UPSITBK4")
    val upsit = upsit0.filter(!(upsit0("UPSITBK1")===""||upsit0("UPSITBK2")===""||upsit0("UPSITBK3")===""||upsit0("UPSITBK4")==="")).cache().
      map(s=>(s(0).toString, s(1).toString, List(s(2), s(3), s(4), s(5)).
      map(_.toString).map(_.toDouble).sum))


    val rem0 = CSVUtils.loadCSVAsTable(sqlContext,"data/REM_Sleep_Disorder_Questionnaire.csv").
      toDF().cache()
    //val remFeatures = rem0.columns.slice(7,28)
    val rem = rem0.rdd.map(s=>(s(2).toString, s(3).toString, List(s(7), s(8), s(9), s(10), s(11), s(12), s(13), s(14), s(15), s(16), s(17), s(18), s(19), s(20), s(21), s(22), s(23), s(24), s(25), s(26), s(27), s(27)).map(_.toString))).
      filter(s=> ! s._3.contains("")).map(s=>(s._1, s._2, s._3.map(_.toInt).sum))

    val moCA = CSVUtils.loadCSVAsTable(sqlContext,"data/Montreal_Cognitive_Assessment__MoCA_.csv").
      toDF().select("PATNO","EVENT_ID","MCATOT").cache()


    val updrsI10 = CSVUtils.loadCSVAsTable(sqlContext,"data/MDS_UPDRS_Part_I__Patient_Questionnaire.csv").
      toDF().cache()
    val updrsI1 = rem0.rdd.map(s=>(s(2).toString, s(3).toString, List(s(7), s(8), s(9), s(10), s(11), s(12), s(13)).map(_.toString))).
      filter(s=> ! s._3.contains("")).map(s=>(s._1, s._2, s._3.map(_.toInt).sum)).filter(s=>s._2=="BL").map(s=>(s._1, s._3))

    val updrsI20 = CSVUtils.loadCSVAsTable(sqlContext,"data/MDS_UPDRS_Part_I.csv").
      toDF().cache()
    val updrsI2 = rem0.rdd.map(s=>(s(2).toString, s(3).toString, List(s(7), s(8), s(9), s(10), s(11), s(12)).map(_.toString))).
      filter(s=> ! s._3.contains("")).map(s=>(s._1, s._2, s._3.map(_.toInt).sum)).filter(s=>s._2=="BL").map(s=>(s._1, s._3))

    val updrsI = updrsI1.join(updrsI2).map(s=>(s._1, s._2._1+s._2._2))

    val updrsII0 = CSVUtils.loadCSVAsTable(sqlContext,"data/MDS_UPDRS_Part_II__Patient_Questionnaire.csv").
      toDF().cache()
    val updrsII = rem0.rdd.map(s=>(s(2).toString, s(3).toString, List(s(7), s(8), s(9), s(10), s(11), s(12), s(13), s(14), s(15), s(16), s(17), s(18), s(19)).map(_.toString))).
      filter(s=> ! s._3.contains("")).map(s=>(s._1, s._2, s._3.map(_.toInt).sum)).filter(s=>s._2=="BL").map(s=>(s._1, s._3))


    val dateFormat01 = new SimpleDateFormat("MM/yyyy")
    val dateFormat02 = new SimpleDateFormat("yyyy")
    val bioChemFeatures = Source.fromFile("data/All_for_filter.txt").getLines().map(_.toLowerCase).toSet[String]

    //to RDD format
    val socEcoRDD = socEco.filter(socEco("EDUCYRS")!=="").map(s=>EDU(s(0).toString, s(1).toString.toInt)).filter(s=>allPatientsID.contains(s.patientID))
    val scDemogrEnrolledRDD = scDemogrEnrolled.filter(!(scDemogrEnrolled("BIRTHDT") === "" || scDemogrEnrolled("PRJENRDT")==="")).
      map(s=>DEMOGR(s(0).toString,s(3).toString, (dateFormat01.parse(s(4).asInstanceOf[String]).getTime - dateFormat02.parse(s(2).asInstanceOf[String]).getTime)/1000/60/60/24/365.0)).
      filter(s=>allPatientsID.contains(s.patientID)).filter(s=>s.age>0)

    val bioChemRDD = bioChemFilterd.map(s=>BIOCHEM(s(0).toString,s(3).toString,s(5).toString.toLowerCase(),s(6))).
      filter(s=>bioChemFeatures.contains(s.testName)).
      filter(s=>s.eventID=="BL"||s.eventID=="SC").
      filter(s=>allPatientsID.contains(s.patientID))

    val datScanRDD = datScan.map(s=>DATSCAN(s._1,s._2,s._3,s._4)).filter(s=>allPatientsID.contains(s.patientID))
    val upsitRDD = upsit.map(s=>UPSIT(s._1,s._2,s._3)).filter(s=>allPatientsID.contains(s.patientID))
    val remRDD = rem.map(s=>REM(s._1,s._2,s._3)).filter(s=>allPatientsID.contains(s.patientID)).filter(s=>s.eventID=="BL"||s.eventID=="SC")

    val moCARDD = moCA.filter(moCA("MCATOT")!== "").map(s=>MOCA(s(0).toString,s(1).toString, s(2).toString.toInt)).
      filter(s=>s.eventID=="BL"||s.eventID=="SC").
      filter(s=>allPatientsID.contains(s.patientID))

    val updrsIRDD = updrsI.map(s=>UPDRS(s._1,s._2)).filter(s=>allPatientsID.contains(s.patientID))
    val updrsIIRDD = updrsII.map(s=>UPDRS(s._1,s._2)).filter(s=>allPatientsID.contains(s.patientID))





    // statistics for Proposal

    val patientHC = patientsWithLabel.filter(s=>s._2==0).map(s=>s._1).collect()
    val patientPD = patientsWithLabel.filter(s=>s._2==1).map(s=>s._1).collect()

    println("HC, PD")
    println(patientHC.length, patientPD.length)

    // Educations
    val socEcoRDDHC = socEcoRDD.filter(s=>patientHC.contains(s.patientID))
    val meanEduHC = socEcoRDDHC.map(_.years).sum()/socEcoRDDHC.map(_.years).count()
    val maxEduHC = socEcoRDDHC.map(_.years).max()
    val minEduHC = socEcoRDDHC.map(_.years).min()

    val socEcoRDDPD = socEcoRDD.filter(s=>patientPD.contains(s.patientID))
    val meanEduPD = socEcoRDDPD.map(_.years).sum()/socEcoRDDPD.map(_.years).count()
    val maxEduPD = socEcoRDDPD.map(_.years).max()
    val minEduPD = socEcoRDDPD.map(_.years).min()

    println("educations")
    println(meanEduHC,maxEduHC,minEduHC)
    println(meanEduPD,maxEduPD,minEduPD)

    //Age
    val scDemogrEnrolledRDDHC = scDemogrEnrolledRDD.filter(s=>patientHC.contains(s.patientID))
    val meanAgeHC = scDemogrEnrolledRDDHC.map(_.age).sum()/scDemogrEnrolledRDDHC.map(_.age).count()
    val maxAgeHC = scDemogrEnrolledRDDHC.map(_.age).max()
    val minAgeHC = scDemogrEnrolledRDDHC.map(_.age).min()

    val scDemogrEnrolledRDDPD = scDemogrEnrolledRDD.filter(s=>patientPD.contains(s.patientID))
    val meanAgePD = scDemogrEnrolledRDDPD.map(_.age).sum()/scDemogrEnrolledRDDPD.map(_.age).count()
    val maxAgePD = scDemogrEnrolledRDDPD.map(_.age).max()
    val minAgePD = scDemogrEnrolledRDDPD.map(_.age).min()


    println("ages")
    println(meanAgeHC,maxAgeHC,minAgeHC)
    println(meanAgePD,maxAgePD,minAgePD)

    println("ERR")
    scDemogrEnrolledRDDPD.filter(s=>s.age<0).take(100).foreach(println)

    //Gender
    val femaleHC = scDemogrEnrolledRDDHC.filter(s=>s.gender == "0" || s.gender == "1").count()
    val maleHC = scDemogrEnrolledRDDHC.filter(s=>s.gender == "2").map(s=>1.0).count()

    val femalePD = scDemogrEnrolledRDDPD.filter(s=>s.gender == "0" || s.gender == "1").count()
    val malePD = scDemogrEnrolledRDDPD.filter(s=>s.gender == "2").count()

    println("gender")
    println(femaleHC,maleHC)
    println(femalePD,malePD)

    //upsit
    val upsitRDDHC = upsitRDD.filter(s=>patientHC.contains(s.patientID))
    val meanUpsitHC = upsitRDDHC.map(_.Score).sum()/upsitRDDHC.map(_.Score).count()
    val maxUpsitHC = upsitRDDHC.map(_.Score).max()
    val minUpsitHC = upsitRDDHC.map(_.Score).min()

    val upsitRDDPD = upsitRDD.filter(s=>patientPD.contains(s.patientID))
    val meanUpsitPD = upsitRDDPD.map(_.Score).sum()/upsitRDDPD.map(_.Score).count()
    val maxUpsitPD = upsitRDDPD.map(_.Score).max()
    val minUpsitPD = upsitRDDPD.map(_.Score).min()

    println("upsit")
    println(meanUpsitHC,maxUpsitHC,minUpsitHC)
    println(meanUpsitPD,maxUpsitPD,minUpsitPD)


    //REM
    val remRDDHC = remRDD.filter(s=>patientHC.contains(s.patientID))
    val meanRemHC = remRDDHC.map(_.Score).sum()/remRDDHC.map(_.Score).count()
    val maxRemHC = remRDDHC.map(_.Score).max()
    val minRemHC = remRDDHC.map(_.Score).min()

    val remRDDPD = remRDD.filter(s=>patientPD.contains(s.patientID))
    val meanRemPD = remRDDPD.map(_.Score).sum()/remRDDPD.map(_.Score).count()
    val maxRemPD = remRDDPD.map(_.Score).max()
    val minRemPD = remRDDPD.map(_.Score).min()

    println("rem")
    println(meanRemHC,maxRemHC,minRemHC)
    println(meanRemPD,maxRemPD,minRemPD)


    //datScan
    val datScanRDDHC = datScanRDD.filter(s=>patientHC.contains(s.patientID))
    val meanCAUDATEHC = datScanRDDHC.map(_.CAUDATE).sum()/datScanRDDHC.map(_.CAUDATE).count()
    val maxCAUDATEHC = datScanRDDHC.map(_.CAUDATE).max()
    val minCAUDATEHC = datScanRDDHC.map(_.CAUDATE).min()

    val meanPUTAMENHC = datScanRDDHC.map(_.PUTAMEN).sum()/datScanRDDHC.map(_.PUTAMEN).count()
    val maxPUTAMENHC = datScanRDDHC.map(_.PUTAMEN).max()
    val minPUTAMENHC = datScanRDDHC.map(_.PUTAMEN).min()

    val datScanRDDPD = datScanRDD.filter(s=>patientPD.contains(s.patientID))
    val meanCAUDATEPD = datScanRDDPD.map(_.CAUDATE).sum()/datScanRDDPD.map(_.CAUDATE).count()
    val maxCAUDATEPD = datScanRDDPD.map(_.CAUDATE).max()
    val minCAUDATEPD = datScanRDDPD.map(_.CAUDATE).min()

    val meanPUTAMENPD = datScanRDDPD.map(_.PUTAMEN).sum()/datScanRDDPD.map(_.PUTAMEN).count()
    val maxPUTAMENPD = datScanRDDPD.map(_.PUTAMEN).max()
    val minPUTAMENPD = datScanRDDPD.map(_.PUTAMEN).min()

    println("datScan CAUDATE")
    println(meanCAUDATEHC,maxCAUDATEHC,minCAUDATEHC)
    println(meanCAUDATEPD,maxCAUDATEPD,minCAUDATEPD)

    println("datScan PUTAMEN")
    println(meanPUTAMENHC,maxPUTAMENHC,minPUTAMENHC)
    println(meanPUTAMENPD,maxPUTAMENPD,minPUTAMENPD)


    //moCA
    val moCARDDHC = moCARDD.filter(s=>patientHC.contains(s.patientID))
    val meanMoCAHC = moCARDDHC.map(_.Score).sum()/moCARDDHC.map(_.Score).count()
    val maxMoCAHC = moCARDDHC.map(_.Score).max()
    val minMoCAHC = moCARDDHC.map(_.Score).min()

    val moCARDDPD = moCARDD.filter(s=>patientPD.contains(s.patientID))
    val meanMoCAPD = moCARDDPD.map(_.Score).sum()/moCARDDPD.map(_.Score).count()
    val maxMoCAPD = moCARDDPD.map(_.Score).max()
    val minMoCAPD = moCARDDPD.map(_.Score).min()

    println("moCA")
    println(meanMoCAHC,maxMoCAHC,minMoCAHC)
    println(meanMoCAPD,maxMoCAPD,minMoCAPD)

    //updrsI
    val updrsIRDDHC = updrsIRDD.filter(s=>patientHC.contains(s.patientID))
    val meanUpdrsIHC = updrsIRDDHC.map(_.Score).sum()/updrsIRDDHC.map(_.Score).count()
    val maxUpdrsIHC = updrsIRDDHC.map(_.Score).max()
    val minUpdrsIHC = updrsIRDDHC.map(_.Score).min()

    val updrsIRDDPD = updrsIRDD.filter(s=>patientPD.contains(s.patientID))
    val meanUpdrsIPD = updrsIRDDPD.map(_.Score).sum()/updrsIRDDPD.map(_.Score).count()
    val maxUpdrsIPD = updrsIRDDPD.map(_.Score).max()
    val minUpdrsIPD = updrsIRDDPD.map(_.Score).min()

    println("updrsI")
    println(meanUpdrsIHC,maxUpdrsIHC,minUpdrsIHC)
    println(meanUpdrsIPD,maxUpdrsIPD,minUpdrsIPD)


    //updrsI
    val updrsIIRDDHC = updrsIIRDD.filter(s=>patientHC.contains(s.patientID))
    val meanUpdrsIIHC = updrsIIRDDHC.map(_.Score).sum()/updrsIIRDDHC.map(_.Score).count()
    val maxUpdrsIIHC = updrsIIRDDHC.map(_.Score).max()
    val minUpdrsIIHC = updrsIIRDDHC.map(_.Score).min()

    val updrsIIRDDPD = updrsIIRDD.filter(s=>patientPD.contains(s.patientID))
    val meanUpdrsIIPD = updrsIIRDDPD.map(_.Score).sum()/updrsIIRDDPD.map(_.Score).count()
    val maxUpdrsIIPD = updrsIIRDDPD.map(_.Score).max()
    val minUpdrsIIPD = updrsIIRDDPD.map(_.Score).min()

    println("updrsII")
    println(meanUpdrsIIHC,maxUpdrsIIHC,minUpdrsIIHC)
    println(meanUpdrsIIPD,maxUpdrsIIPD,minUpdrsIIPD)


    (socEcoRDD, scDemogrEnrolledRDD, bioChemRDD, datScanRDD, upsitRDD, remRDD, moCARDD, updrsIRDD, updrsIIRDD, patientsWithLabel)
  }

  def createContext: SparkContext = {
    val conf = new SparkConf().setAppName("CSE 8803 Homework Two Application").setMaster("local")
    new SparkContext(conf)
  }
}
