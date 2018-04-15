/**
 * @author Hang Su <hangsu@gatech.edu>.
 */

package edu.gatech.cse8803.main

import java.text.{NumberFormat, SimpleDateFormat}

import edu.gatech.cse8803.clustering.{Metrics, NMF}
import edu.gatech.cse8803.features.FeatureConstruction
import edu.gatech.cse8803.ioutils.CSVUtils
import edu.gatech.cse8803.model._
import edu.gatech.cse8803.statistics.statistics.printStatistics
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
    val (socEcoRDD, scDemogrRDD, famHistRDD, bioChemRDD, datScanRDD,
    quipRDD, hvltRDD, bLineRDD, semaFlutRDD, lnSeqRDD, sdModRDD, essRDD, staiRDD, gdsRDD, scopaRDD,
    upsitRDD, remRDD, moCARDD, updrsIRDD, updrsIIRDD, updrsIIIRDD, patientsWithLabel) = loadRddRawData(sqlContext)

    /** calculate and print statistics*/
    // printStatistics(socEcoRDD, scDemogrRDD, bioChemRDD, datScanRDD, quipRDD, upsitRDD, remRDD, moCARDD, updrsIRDD, updrsIIRDD, patientsWithLabel)

    /** conduct phenotyping */
    val phenotypeLabel = patientsWithLabel


    /** feature construction with all features except UPDRS */
    val featureTuplesWithoutUPDRS = sc.union(
      FeatureConstruction.constructSocEcoFeatureTuple(socEcoRDD),
      FeatureConstruction.constructScDemogrFeatureTuple(scDemogrRDD),
      FeatureConstruction.constructFamHistFeatureTuple(famHistRDD), // New
      FeatureConstruction.constructBioChemRDDFeatureTuple(bioChemRDD),
      FeatureConstruction.constructDatScanRDDFeatureTuple(datScanRDD),

      FeatureConstruction.constructQuipRDDFeatureTuple(quipRDD), //new
      FeatureConstruction.constructHvltRDDFeatureTuple(hvltRDD), //new
      FeatureConstruction.constructBLineRDDFeatureTuple(bLineRDD), //new
      FeatureConstruction.constructSemaFlutRDDFeatureTuple(semaFlutRDD), //new
      FeatureConstruction.constructLnSeqRDDFeatureTuple(lnSeqRDD), //new
      FeatureConstruction.constructSdModRDDFeatureTuple(sdModRDD), //new
      FeatureConstruction.constructEssRDDFeatureTuple(essRDD), //new
      FeatureConstruction.constructStaiRDDFeatureTuple(staiRDD), //new
      FeatureConstruction.constructGdsRDDFeatureTuple(gdsRDD), //new
      FeatureConstruction.constructScopaRDDFeatureTuple(scopaRDD), //new

      FeatureConstruction.constructUpsitRDDFeatureTuple(upsitRDD),
      FeatureConstruction.constructRemRDDFeatureTuple(remRDD),
      FeatureConstruction.constructMoCARDDFeatureTuple(moCARDD)
    )

    /** feature construction with all features */
    val featureTuplesWithUPDRS = sc.union(
      featureTuplesWithoutUPDRS,
      FeatureConstruction.constructUpdrsIRDDFeatureTuple(updrsIRDD),
      FeatureConstruction.constructUpdrsIIRDDFeatureTuple(updrsIIRDD),
      FeatureConstruction.constructUpdrsIIIRDDFeatureTuple(updrsIIIRDD) // New
    )


//    val rawFeaturesWithUPDRS = FeatureConstruction.construct(sc, featureTuplesWithUPDRS, phenotypeLabel)
//    val rawFeaturesWithoutUPDRS = FeatureConstruction.construct(sc, featureTuplesWithoutUPDRS, phenotypeLabel)

//    val rawDenseFeaturesWithUPRDS = FeatureConstruction.constructDense(sc, featureTuplesWithUPDRS, phenotypeLabel)

//    FeatureConstruction.saveDenseFeatures(sc, rawDenseFeaturesWithUPRDS, phenotypeLabel, 1)


//
    FeatureConstruction.saveFeatures(sc, featureTuplesWithUPDRS, phenotypeLabel, 1) // withUPDRS == 1
    FeatureConstruction.saveFeatures(sc, featureTuplesWithoutUPDRS, phenotypeLabel, 0)  // withoutUPDRS == 0


    // val (kMeansPurity, gaussianMixturePurity, streamKmeansPurity, nmfPurity) = testClustering(phenotypeLabel, rawFeaturesWithoutUPDRS)
    // val (kMeansPurity, gaussianMixturePurity, streamKmeansPurity, nmfPurity) = testClustering(phenotypeLabel, rawFeaturesWithUPDRS)


/*    println(f"[All feature] purity of kMeans is: $kMeansPurity%.5f")
    println(f"[All feature] purity of GMM is: $gaussianMixturePurity%.5f")
    println(f"[All feature] purity of StreamingKMeans is: $streamKmeansPurity%.5f")
    println(f"[All feature] purity of NMF is: $nmfPurity%.5f")*/

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


/*    val kMeansPurity = 0.0
    val gaussianMixturePurity = 0.0
    val streamKmeansPurity = 0.0*/
    val nmfPurity = 0.0

    (kMeansPurity, gaussianMixturePurity, streamKmeansPurity, nmfPurity)
  }


  def loadRddRawData(sqlContext: SQLContext): (RDD[EDU], RDD[DEMOGR], RDD[HIST], RDD[BIOCHEM], RDD[DATSCAN],
    RDD[QUIP], RDD[HVLT], RDD[BLINE], RDD[SFT],RDD[LNSEQ], RDD[SDMOD],RDD[ESS], RDD[STAI],RDD[GDS], RDD[SCOPA],
    RDD[UPSIT], RDD[REM], RDD[MOCA], RDD[UPDRS], RDD[UPDRS], RDD[UPDRS], RDD[(String, Int)]) = {
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


    /**_Subject_Characteristics*/

    /**Socio-Economics*/
    val socEco = CSVUtils.loadCSVAsTable(sqlContext,"data/Socio-Economics.csv").
      toDF().select("PATNO","EDUCYRS").cache() //Event ID & F_status
    val socEcoRDD = socEco.filter(socEco("EDUCYRS")!=="").map(s=>EDU(s(0).toString, s(1).toString.toInt)).filter(s=>allPatientsID.contains(s.patientID))

    /**Demographics*/
    val scDemogr = CSVUtils.loadCSVAsTable(sqlContext,"data/Screening___Demographics.csv").
      toDF().select("PATNO","APPRDX","BIRTHDT","GENDER","PRJENRDT","DECLINED","EXCLUDED") // "PRJENRDT" is "projected enrollment date"
    val scDemogrEnrolled = scDemogr.filter(!(scDemogr("EXCLUDED") === "1" || scDemogr("DECLINED") === "1")).cache()

    val dateFormat01 = new SimpleDateFormat("MM/yy")
    val dateFormat02 = new SimpleDateFormat("yyyy")

    val scDemogrEnrolledRDD = scDemogrEnrolled.filter(!(scDemogrEnrolled("BIRTHDT") === "" || scDemogrEnrolled("PRJENRDT")==="")).
      map(s=>DEMOGR(s(0).toString,s(3).toString, (dateFormat01.parse(s(4).asInstanceOf[String]).getTime - dateFormat02.parse(s(2).asInstanceOf[String]).getTime)/1000/60/60/24/365.25)).
      filter(s=>allPatientsID.contains(s.patientID)).filter(s=>s.age>0)

    /**Family History*/
    val famHist = CSVUtils.loadCSVAsTable(sqlContext,"data/Family_History__PD_.csv").
      toDF().cache().rdd.map(s=>(s(2).toString, s(3).toString,
      List(s(6), s(8), s(10), s(12), s(14), s(16), s(18), s(20), s(22)).map(_.toString()).map{ case "" => "0"; case x => x}, // all relatives
      List(s(7), s(9), s(11), s(13), s(15), s(17), s(19), s(21), s(23)).map(_.toString()).map{ case "" => "0"; case x => x})). // with PDs
      map(s=>(s._1, s._2, s._3.map(_.toInt).sum, s._4.map(_.toDouble).sum)).
      filter(s=> s._3 != 0).
      map(s=>(s._1, s._2, s._4/s._3))

    val famHistRDD = famHist.map(s=>HIST(s._1,s._2,s._3)).
      filter(s=>allPatientsID.contains(s.patientID)).
      filter(s=>s.eventID=="SC")


    /**_Biospecimen*/

    val bioChem = CSVUtils.loadCSVAsTable(sqlContext,"data/Biospecimen_Analysis_Results.csv").
      toDF().select("PATNO","GENDER","DIAGNOSIS","CLINICAL_EVENT","TYPE","TESTNAME","TESTVALUE", "UNITS").cache() //DNA RNA biochemical
    val bioChemFilterd = bioChem.filter((bioChem("TESTVALUE") !== "") and bioChem("TESTVALUE").isNotNull
      and (bioChem("TESTVALUE")!== "Undetermined") and (bioChem("TESTVALUE") !== "N/A") and (bioChem("TESTVALUE") !== "NA"))

    /** All features in Biospecimen*/

    val bioChemRDD = bioChemFilterd.map(s=>BIOCHEM(s(0).toString,s(3).toString, s(4).toString.toLowerCase(), s(5).toString.toLowerCase(),s(6).toString, s(7).toString)).
      filter(s=>s.eventID=="BL"||s.eventID=="SC").
      filter(s=>allPatientsID.contains(s.patientID))

/*    bioChemRDD.map(s=>((s.testType, s.testName), s.value.toString)).filter(s=> ! s._2.matches("[-+]?[0-9]*\\.?[0-9]*")).
      distinct.groupByKey().sortByKey().collect().foreach(println)*/


    /** Features selected base on literature */
/*    val bioChemFeatures = Source.fromFile("data/All_for_filter.txt").getLines().map(_.toLowerCase).toSet[String]

    val bioChemRDD = bioChemFilterd.map(s=>BIOCHEM(s(0).toString,s(3).toString, s(4).toString.toLowerCase(), s(5).toString.toLowerCase(), s(6), s(7).toString)).
      filter(s=>bioChemFeatures.contains(s.testName)).
      filter(s=>s.eventID=="BL"||s.eventID=="SC").
      filter(s=>allPatientsID.contains(s.patientID))*/





    /**Imaging*/

    /**DATScan_Analysis*/
    val datScan = CSVUtils.loadCSVAsTable(sqlContext,"data/DATScan_Analysis.csv").
      toDF().select("PATNO","EVENT_ID","CAUDATE_R","CAUDATE_L","PUTAMEN_R","PUTAMEN_L").cache().
      map{s=>
        val caudate = List(s(2), s(3)).map(_.toString.toDouble).sum
        val putamen = List(s(4), s(5)).map(_.toString.toDouble).sum
        val caudateAsym = 100*(s(2).toString.toDouble - s(3).toString.toDouble)/caudate
        val putamenAsym = 100*(s(4).toString.toDouble - s(5).toString.toDouble)/putamen
        (s(0).toString, s(1).toString, caudate, putamen, caudate/putamen, caudateAsym, putamenAsym)}

    val datScanRDD = datScan.map(s=>DATSCAN(s._1,s._2,s._3,s._4,s._5,s._6,s._7)).filter(s=>allPatientsID.contains(s.patientID)).filter(s=>s.eventID=="SC")


    /**Non-motor_Assessments*/

    /**Questionnaire for Impulsive-Compulsive Disorders in Parkinson’s Disease*/
    val quip = CSVUtils.loadCSVAsTable(sqlContext,"data/QUIP_Current_Short.csv").
      toDF().cache().rdd.map(s=>(s(2).toString, s(3).toString,
      List(s(7), s(8)).map(_.toString),
      List(s(9), s(10)).map(_.toString),
      List(s(11), s(12)).map(_.toString),
      List(s(13), s(14)).map(_.toString),
      List(s(15), s(16), s(17)).map(_.toString))).
      filter(s=> ! (s._3.contains("") || s._4.contains("") || s._5.contains("") || s._6.contains("") || s._7.contains(""))).
      map(s=> (s._1, s._2,
        if (s._3.map(_.toInt).sum>0) 1 else 0,
        if (s._4.map(_.toInt).sum>0) 1 else 0,
        if (s._5.map(_.toInt).sum>0) 1 else 0,
        if (s._6.map(_.toInt).sum>0) 1 else 0,
        s._7.map(_.toInt).sum)).
      map(s=> (s._1, s._2, s._3 + s._4 + s._5 + s._6 + s._7))

    val quipRDD = quip.map(s=>QUIP(s._1,s._2,s._3)).filter(s=>allPatientsID.contains(s.patientID))


    /*Hopkins_Verbal_Learning_Test*/
    val hvlt = CSVUtils.loadCSVAsTable(sqlContext,"data/Hopkins_Verbal_Learning_Test.csv").
      toDF().select("PATNO", "EVENT_ID", "HVLTRT1", "HVLTRT2", "HVLTRT3", "HVLTRDLY", "HVLTREC", "HVLTFPRL", "HVLTFPUN").
      cache().rdd.filter(s=> ! List(s(2), s(3), s(4), s(5), s(6), s(7), s(8)).contains("")).
      map(s=>(s(0).toString, s(1).toString,
        List(s(2), s(3), s(4)).map(_.toString.toInt).sum,
        s(6).toString.toInt - s(7).toString.toInt - s(8).toString.toInt,
        s(5).toString.toInt/List(s(3), s(4)).map(_.toString.toInt).max))

    val hvltRDD = hvlt.map(s=>HVLT(s._1,s._2,s._3,s._4, s._5)).
      filter(s=>allPatientsID.contains(s.patientID)).
      filter(s=>s.eventID=="BL")


    /*Benton_Judgment_of_Line_Orientation
    * Sum of BJLOT1 - BJLOT30*/

    val bLine = CSVUtils.loadCSVAsTable(sqlContext,"data/Benton_Judgment_of_Line_Orientation.csv").
      toDF().select("PATNO", "EVENT_ID","DVS_JLO_MSSAE").cache().rdd. // Derived-MOANS (Age and Education)
      map(s=>(s(0).toString, s(1).toString, s(2).toString)).filter(s=> !(s._3==""))

    val bLineRDD = bLine.map(s=>BLINE(s._1, s._2, s._3.toDouble)).
      filter(s=>allPatientsID.contains(s.patientID)).
      filter(s=>s.eventID=="BL")


    /**Semantic_Fluency*/
    val semaFlut = CSVUtils.loadCSVAsTable(sqlContext,"data/Semantic_Fluency.csv").
      toDF().select("PATNO", "EVENT_ID", "VLTANIM", "VLTVEG", "VLTFRUIT", "DVS_SFTANIM", "DVT_SFTANIM").cache().rdd.
      filter(s=> ! List(s(2), s(3), s(4), s(5), s(6)).contains("")).
      map(s=>(s(0).toString, s(1).toString, List(s(2), s(3), s(4)).map(_.toString.toInt).sum, s(5).toString.toInt, s(6).toString.toInt))

    val semaFlutRDD = semaFlut.map(s=>SFT(s._1, s._2, s._3, s._4, s._5)).
      filter(s=>allPatientsID.contains(s.patientID)).
      filter(s=>s.eventID=="BL")


    /** Letter Number Sequencing */
    val lnSeq = CSVUtils.loadCSVAsTable(sqlContext,"data/Letter_-_Number_Sequencing__PD_.csv").
      toDF().select("PATNO", "EVENT_ID","LNS_TOTRAW", "DVS_LNS").cache().rdd.
      filter(s=> ! List(s(2), s(3)).contains("")).
      map(s=>(s(0).toString, s(1).toString, s(2).toString.toInt, s(3).toString.toInt))

    val lnSeqRDD = lnSeq.map(s=>LNSEQ(s._1, s._2, s._3, s._4)).
      filter(s=>allPatientsID.contains(s.patientID)).
      filter(s=>s.eventID=="BL")


    /** Symbol_Digit_Modalities */
    val sdMod = CSVUtils.loadCSVAsTable(sqlContext,"data/Symbol_Digit_Modalities.csv").
      toDF().select("PATNO", "EVENT_ID","SDMTOTAL", "DVT_SDM").cache().rdd.
      filter(s=> ! List(s(2), s(3)).contains("")).
      map(s=>(s(0).toString, s(1).toString, s(2).toString.toInt, s(3).toString.toDouble))

    val sdModRDD = sdMod.map(s=>SDMOD(s._1, s._2, s._3, s._4)).
      filter(s=>allPatientsID.contains(s.patientID)).
      filter(s=>s.eventID=="BL")


    /** Epworth_Sleepiness_Scale
      * Sum of ESS1 - ESS8.
      * Subjects with ESS <10 are "Not Sleepy".
      * Subjects with ESS >=10 are "Sleepy".*/

    val ess = CSVUtils.loadCSVAsTable(sqlContext,"data/Epworth_Sleepiness_Scale.csv").
      toDF().cache().rdd.
      map(s=>(s(2).toString, s(3).toString, List(s(7), s(8), s(9), s(10), s(11), s(12), s(13), s(14)).map(_.toString()))).
      filter(s=> ! s._3.contains("")).
      map(s=>(s._1, s._2, s._3.map(_.toInt).sum))

    val essRDD = ess.map(s=>ESS(s._1,s._2,s._3)).filter(s=>allPatientsID.contains(s.patientID)).
      filter(s=>s.eventID=="BL")


    /** State-Trait_Anxiety_Inventory
      *
      * STAI - State Subscore
      * STAIAD1 - STAIAD20.  Add values for the following questions:  3, 4, 6, 7, 9, 12, 13, 14, 17, 18.
      * Use reverse scoring for the values of the remaining questions through question 20 and add to the first value.
      *
      * STAI - Trait Subscore
      * STAIAD21 - STAIAD40.  Add values for the following questions:  22, 24, 25, 28, 29, 31, 32, 35, 37, 38, 40.
      * Use reverse scoring for the values of the remaining questions and add to the first value.*/

    val stai = CSVUtils.loadCSVAsTable(sqlContext,"data/State-Trait_Anxiety_Inventory.csv").
      toDF().cache().rdd.map(s=>(s(2).toString, s(3).toString,
      List(s(8), s(9), s(11), s(12), s(14), s(17), s(18), s(19), s(22), s(23)).map(_.toString()),
      List(s(6), s(7), s(10), s(13), s(15), s(16), s(20), s(21), s(24), s(25)).map(_.toString()),
      List(s(27), s(29), s(30), s(33), s(34), s(36), s(37), s(40), s(42), s(43), s(45)).map(_.toString()),
      List(s(26), s(28), s(31), s(32), s(35), s(38), s(39), s(41), s(44)).map(_.toString()))).
      filter(s=> ! (s._3.contains("")||s._4.contains("")||s._5.contains("")||s._6.contains(""))).
      map(s=>(s._1, s._2,
        s._3.map(_.toInt).sum + s._4.length*5 - s._4.map(_.toInt).sum,
        s._5.map(_.toInt).sum + s._6.length*5 - s._6.map(_.toInt).sum))

    val staiRDD = stai.map(s=>STAI(s._1,s._2,s._3+s._4, s._3, s._4)).filter(s=>allPatientsID.contains(s.patientID)).
      filter(s=>s.eventID=="BL")


    /** Geriatric Depression Scale (GDS)
      * Add 1 point for each response of "No" (0) to any of the following variables:
      * GDSSATIS, GDSGSPIR, GDSHAPPY, GDSALIVE, GDSENRGY.
      * Add 1 point for each response of "Yes" (1) to any of the following variables:
      * GDSDROPD, GDSEMPTY, GDSBORED, GDSAFRAD, GDSHLPLS, GDSHOME, GDSMEMRY, GDSWRTLS, GDSHOPLS, GDSBETER. */

    val gds = CSVUtils.loadCSVAsTable(sqlContext,"data/Geriatric_Depression_Scale__Short_.csv").
      toDF().cache().rdd.map(s=>(s(2).toString, s(3).toString,
      List(s(7), s(8), s(9), s(11), s(13), s(14), s(15), s(17), s(19), s(20)).map(_.toString()),
      List(s(6), s(10),s(12), s(16), s(18)).map(_.toString()))).
      filter(s=> ! (s._3.contains("")||s._4.contains(""))).
      map(s=>(s._1, s._2,
        s._3.map(_.toInt).sum,
        s._4.length - s._4.map(_.toInt).sum))

    val gdsRDD = gds.map(s=>GDS(s._1,s._2, s._3 + s._4)).filter(s=>allPatientsID.contains(s.patientID)).
      filter(s=>s.eventID=="BL")


    /** SCOPA-AUT Total Score
      * Scales for Outcomes in Parkinson's disease – Autonomic
      * SCAU1 - SCAU25.
      * For questions 1-21 (SCAU1 - SCAU21), add 3 points for each response of "9". Otherwise, add the number of points in response.
      * For questions 22-25 (SCAU22 - SCAU25), add 0 points for each response of "9". Otherwise, add the number of points in response.
      *
      * Gastrointestinal – questions # 1-7
        Urinary questions #8-13 (if they use a catheter, they get the highest score)
        Cardiovascular -questions # 14-16
        Thermoregulatory - questions # 17-18, 20-21
        Pupillomotor - question #19
        Sexual - questions #22-23 if male or - questions # 24-25 if female
        (Questions #23a and #26 do not contribute to the score)*/

    val scopa = CSVUtils.loadCSVAsTable(sqlContext,"data/SCOPA-AUT.csv").
      toDF().cache().rdd.
      map(s=>(s(2).toString, s(3).toString,
      List(s(7), s(8), s(9), s(10), s(11), s(12), s(13)).map(_.toString()),
      List(s(14), s(15), s(16), s(17), s(18), s(19)).map(_.toString()).map { case "9" => "3"; case x => x },
      List(s(20), s(21), s(22)).map(_.toString()),
      List(s(23), s(24), s(26), s(27)).map(_.toString()),
      s(25).toString,
      List(s(28), s(29)).map(_.toString()).map { case "9" => "0"; case x => x },
      List(s(32), s(33)).map(_.toString()).map { case "9" => "0"; case x => x })).
      filter(s=> ! (s._3.contains("")||s._4.contains("")||s._5.contains("")||s._6.contains("")||s._7.contains("")||
        (s._8.contains("") && s._9.contains("")))).
      map(s=>(s._1, s._2,
        s._3.map(_.toInt).sum,
        s._4.map(_.toInt).sum,
        s._5.map(_.toInt).sum,
        s._6.map(_.toInt).sum,
        s._7.toInt,
        if (s._9.contains("")) s._8.map(_.toInt).sum else s._9.map(_.toInt).sum)).
      map(s=>(s._1, s._2, List(s._3, s._4, s._5, s._6, s._7, s._8).sum))

    val scopaRDD = scopa.map(s=>SCOPA(s._1, s._2, s._3)).filter(s=>allPatientsID.contains(s.patientID)).
      filter(s=>s.eventID=="BL")


/*    val upsit0 = CSVUtils.loadCSVAsTable(sqlContext,"data/University_of_Pennsylvania_Smell_ID_Test.csv").
      toDF().select("PATNO","EVENT_ID","UPSITBK1","UPSITBK2","UPSITBK3","UPSITBK4")
    val upsit = upsit0.filter(!(upsit0("UPSITBK1")===""||upsit0("UPSITBK2")===""||upsit0("UPSITBK3")===""||upsit0("UPSITBK4")==="")).cache().
      map(s=>(s(0).toString, s(1).toString, List(s(2), s(3), s(4), s(5)).
      map(_.toString).map(_.toInt).sum))*/


    /**University_of_Pennsylvania_Smell_ID_Test*/
    val upsit = CSVUtils.loadCSVAsTable(sqlContext,"data/University_of_Pennsylvania_Smell_ID_Test.csv").
      toDF().select("PATNO","EVENT_ID","UPSITBK1","UPSITBK2","UPSITBK3","UPSITBK4").cache().rdd.
      map(s=>(s(0).toString, s(1).toString, List(s(2), s(3), s(4), s(5)).map(_.toString))).
      filter(s=> ! s._3.contains("")).
      map(s=>(s._1, s._2, s._3.map(_.toInt).sum))

    val upsitRDD = upsit.map(s=>UPSIT(s._1,s._2,s._3)).filter(s=>allPatientsID.contains(s.patientID))


    /*REM_Sleep_Disorder*/
    val rem = CSVUtils.loadCSVAsTable(sqlContext,"data/REM_Sleep_Disorder_Questionnaire.csv").
      toDF().cache().rdd.
      map(s=>(s(2).toString, s(3).toString, List(s(7), s(8), s(9), s(10), s(11), s(12), s(13), s(14), s(15), s(16),
        s(17), s(18), s(19), s(20), s(21), s(22), s(23), s(24), s(25), s(26), s(27), s(27)).map(_.toString()))).
      filter(s=> ! s._3.contains("")).
      map(s=>(s._1, s._2, s._3.map(_.toInt).sum))

    val remRDD = rem.map(s=>REM(s._1,s._2,s._3)).filter(s=>allPatientsID.contains(s.patientID)).
      filter(s=>s.eventID=="BL"||s.eventID=="SC")


    /*Montreal_Cognitive_Assessment*/
    val moCA = CSVUtils.loadCSVAsTable(sqlContext,"data/Montreal_Cognitive_Assessment__MoCA_.csv").
      toDF().select("PATNO","EVENT_ID","MCATOT").cache()

    val moCARDD = moCA.filter(moCA("MCATOT")!== "").map(s=>MOCA(s(0).toString,s(1).toString, s(2).toString.toInt)).
      filter(s=>s.eventID=="BL"||s.eventID=="SC").
      filter(s=>allPatientsID.contains(s.patientID))




    /**Motor_Assessments*/

    val updrsI1 = CSVUtils.loadCSVAsTable(sqlContext,"data/MDS_UPDRS_Part_I__Patient_Questionnaire.csv").
      toDF().rdd.map(s=>(s(2).toString, s(3).toString, List(s(7), s(8), s(9), s(10), s(11), s(12), s(13)).map(_.toString))).
      filter(s=> ! s._3.contains("")).map(s=>(s._1, s._2, s._3.map(_.toInt).sum)).filter(s=>s._2=="BL").map(s=>(s._1, s._3))

    val updrsI2 = CSVUtils.loadCSVAsTable(sqlContext,"data/MDS_UPDRS_Part_I.csv").
      toDF().cache().rdd.map(s=>(s(2).toString, s(3).toString, List(s(7), s(8), s(9), s(10), s(11), s(12)).map(_.toString))).
      filter(s=> ! s._3.contains("")).map(s=>(s._1, s._2, s._3.map(_.toInt).sum)).filter(s=>s._2=="BL").map(s=>(s._1, s._3))

    val updrsI = updrsI1.join(updrsI2).map(s=>(s._1, s._2._1+s._2._2))

    val updrsII = CSVUtils.loadCSVAsTable(sqlContext,"data/MDS_UPDRS_Part_II__Patient_Questionnaire.csv").
      toDF().cache().rdd.map(s=>(s(2).toString, s(3).toString, List(s(7), s(8), s(9), s(10), s(11), s(12), s(13), s(14), s(15), s(16), s(17), s(18), s(19)).map(_.toString))).
      filter(s=> ! s._3.contains("")).map(s=>(s._1, s._2, s._3.map(_.toInt).sum)).filter(s=>s._2=="BL").map(s=>(s._1, s._3))

    val updrsIII = CSVUtils.loadCSVAsTable(sqlContext,"data/MDS_UPDRS_Part_III__Post_Dose_.csv").
      toDF().cache().rdd.map(s=>(s(2).toString, s(3).toString, List(s(8), s(9), s(10),
      s(11), s(12), s(13), s(14), s(15), s(16), s(17), s(18), s(19), s(20),
      s(21), s(22), s(23), s(24), s(25), s(26), s(27), s(28), s(29), s(30),
      s(31), s(32), s(33), s(34), s(35), s(36), s(37), s(38), s(39), s(40)
    ).map(_.toString))).
      filter(s=> ! s._3.contains("")).map(s=>(s._1, s._2, s._3.map(_.toInt).sum)).filter(s=>s._2=="BL").map(s=>(s._1, s._3))

    val updrsIRDD = updrsI.map(s=>UPDRS(s._1,s._2)).filter(s=>allPatientsID.contains(s.patientID))
    val updrsIIRDD = updrsII.map(s=>UPDRS(s._1,s._2)).filter(s=>allPatientsID.contains(s.patientID))
    val updrsIIIRDD = updrsIII.map(s=>UPDRS(s._1,s._2)).filter(s=>allPatientsID.contains(s.patientID))





    (socEcoRDD, scDemogrEnrolledRDD, famHistRDD, bioChemRDD, datScanRDD,
      quipRDD, hvltRDD, bLineRDD, semaFlutRDD, lnSeqRDD, sdModRDD, essRDD, staiRDD, gdsRDD, scopaRDD,
      upsitRDD, remRDD, moCARDD, updrsIRDD, updrsIIRDD, updrsIIIRDD, patientsWithLabel)

  }






  def createContext: SparkContext = {
    val conf = new SparkConf().setAppName("CSE 8803 Homework Two Application").setMaster("local")
    new SparkContext(conf)
  }
}
