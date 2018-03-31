/**
 * @author Hang Su
 */
package edu.gatech.cse8803.features

import edu.gatech.cse8803.model._
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._


object FeatureConstruction {

  /**
   * ((patient-id, feature-name), feature-value)
   */
  type FeatureTuple = ((String, String), Double)

  /**
   * Aggregate feature tuples from diagnostic with COUNT aggregation,
   * @param socEcoRDD RDD of education years
   * @return RDD of feature tuples
   */
  def constructSocEcoFeatureTuple(socEcoRDD: RDD[EDU]): RDD[FeatureTuple] = {
    val featureTuples = socEcoRDD.map(s=>((s.patientID, "eduYears"),s.years.toDouble))
    featureTuples
  }

  /**
   * Aggregate feature tuples from medication with COUNT aggregation,
   * @param scDemogrRDD RDD of medication
   * @return RDD of feature tuples
   */
  def constructScDemogrFeatureTuple(scDemogrRDD: RDD[DEMOGR]): RDD[FeatureTuple] = {
    val featureTuples_G = scDemogrRDD.map(s=>((s.patientID, "gender"), s.gender.toDouble))
    val featureTuples_A = scDemogrRDD.map(s=>((s.patientID, "age"), s.age))
    featureTuples_G.union(featureTuples_A)
  }


  def constructFamHistFeatureTuple(famHistRDD: RDD[HIST]): RDD[FeatureTuple] = {
    val featureTuples = famHistRDD.map(s=>((s.patientID, "familyHistory"),s.Score))
    featureTuples
  }



  /**
   * Aggregate feature tuples from lab result, using AVERAGE aggregation
   * @param datScanRDD RDD of lab result
   * @return RDD of feature tuples
   */
  def constructDatScanRDDFeatureTuple(datScanRDD: RDD[DATSCAN]): RDD[FeatureTuple] = {
    val featureTuples_C = datScanRDD.map(s=>((s.patientID, "CAUDATE"), s.CAUDATE))
    val featureTuples_P = datScanRDD.map(s=>((s.patientID, "PUTAMEN"), s.PUTAMEN))
    val featureTuples_R = datScanRDD.map(s=>((s.patientID, "CAUDATE_PUTAMEN_ratio"), s.cpRatio))
    val featureTuples_CA = datScanRDD.map(s=>((s.patientID, "CAUDATE_Asym"), s.caudateAsym))
    val featureTuples_PA = datScanRDD.map(s=>((s.patientID, "PUTAMEN_Asym"), s.putamenAsym))
    featureTuples_C.union(featureTuples_P).union(featureTuples_R).union(featureTuples_CA).union(featureTuples_PA)
  }



  def constructQuipRDDFeatureTuple(quipRDD: RDD[QUIP]): RDD[FeatureTuple] = {
    val featureTuples = quipRDD.filter(s=>s.eventID=="BL").map(s=>((s.patientID, "QUIP"), s.Score.toDouble))
    featureTuples
  }

  def constructHvltRDDFeatureTuple(hvltRDD: RDD[HVLT]): RDD[FeatureTuple] = {
    val featureTuples_I = hvltRDD.map(s=>((s.patientID, "HVLT_immediateScore"), s.immediateScore.toDouble))
    val featureTuples_D = hvltRDD.map(s=>((s.patientID, "HVLT_discrimScore"), s.discrimScore.toDouble))
    val featureTuples_R = hvltRDD.map(s=>((s.patientID, "HVLT_retentScore"), s.retentScore))
    featureTuples_I.union(featureTuples_D).union(featureTuples_R)
  }


  def constructBLineRDDFeatureTuple(bLineRDD: RDD[BLINE]): RDD[FeatureTuple] = {
    val featureTuples = bLineRDD.filter(s=>s.eventID=="BL").map(s=>((s.patientID, "Benton_Line_Orientation"), s.Score.toDouble))
    featureTuples
  }

  def constructSemaFlutRDDFeatureTuple(semaFlutRDD: RDD[SFT]): RDD[FeatureTuple] = {
    val featureTuples_S = semaFlutRDD.map(s=>((s.patientID, "SemaFlut_sumScore"), s.sumScore.toDouble))
    val featureTuples_DT = semaFlutRDD.map(s=>((s.patientID, "SemaFlut_derivedScaled"), s.derivedSemScaled.toDouble))
    val featureTuples_DS = semaFlutRDD.map(s=>((s.patientID, "SemaFlut_derivedT"), s.derivedSemT.toDouble))
    featureTuples_S.union(featureTuples_DT).union(featureTuples_DS)
  }

  def constructLnSeqRDDFeatureTuple(lnSeqRDD: RDD[LNSEQ]): RDD[FeatureTuple] = {
    val featureTuples_S = lnSeqRDD.map(s=>((s.patientID, "LnSeq_Score"), s.Score.toDouble))
    val featureTuples_DS = lnSeqRDD.map(s=>((s.patientID, "LnSeq_derivedScore"), s.derivedScore.toDouble))
    featureTuples_S.union(featureTuples_DS)
  }

  def constructSdModRDDFeatureTuple(sdModRDD: RDD[SDMOD]): RDD[FeatureTuple] = {
    val featureTuples_S = sdModRDD.map(s=>((s.patientID, "SdMod_Score"), s.Score.toDouble))
    val featureTuples_DT = sdModRDD.map(s=>((s.patientID, "SdMod_derivedTScore"), s.derivedTScore.toDouble))
    featureTuples_S.union(featureTuples_DT)
  }

  def constructEssRDDFeatureTuple(essRDD: RDD[ESS]): RDD[FeatureTuple] = {
    val featureTuples = essRDD.filter(s=>s.eventID=="BL").map(s=>((s.patientID, "ESS"), s.Score.toDouble))
    featureTuples
  }

  def constructStaiRDDFeatureTuple(staiRDD: RDD[STAI]): RDD[FeatureTuple] = {
    val featureTuples_S = staiRDD.map(s=>((s.patientID, "STAI_Score"), s.Score.toDouble))
    val featureTuples_SS = staiRDD.map(s=>((s.patientID, "STAI_stateScore"), s.stateScore.toDouble))
    val featureTuples_TS = staiRDD.map(s=>((s.patientID, "STAI_traitScore"), s.traitScore.toDouble))
    featureTuples_S.union(featureTuples_SS).union(featureTuples_TS)
  }

  def constructGdsRDDFeatureTuple(gdsRDD: RDD[GDS]): RDD[FeatureTuple] = {
    val featureTuples = gdsRDD.map(s => ((s.patientID, "Gds_Score"), s.Score.toDouble))
    featureTuples
  }

  def constructScopaRDDFeatureTuple(scopaRDD: RDD[SCOPA]): RDD[FeatureTuple] = {
    val featureTuples = scopaRDD.filter(s=>s.eventID=="BL").map(s=>((s.patientID, "SCOPA"), s.Score.toDouble))
    featureTuples
  }



  /**
    * Aggregate feature tuples from lab result, using AVERAGE aggregation
    * @param upsitRDD RDD of lab result
    * @return RDD of feature tuples
    */

  def constructUpsitRDDFeatureTuple(upsitRDD: RDD[UPSIT]): RDD[FeatureTuple] = {
    val featureTuples = upsitRDD.filter(s=>s.eventID=="SC"||s.eventID=="BL").map(s=>((s.patientID, "UPSIT"), s.Score.toDouble))
    featureTuples
  }


  /**
    * Aggregate feature tuples from lab result, using AVERAGE aggregation
    * @param remRDD RDD of lab result
    * @return RDD of feature tuples
    */

  def constructRemRDDFeatureTuple(remRDD: RDD[REM]): RDD[FeatureTuple] = {
    val featureTuples = remRDD.map(s=>((s.patientID, "REM"), s.Score.toDouble))
    featureTuples
  }

  /**
    * Aggregate feature tuples from lab result, using AVERAGE aggregation
    * @param moCARDD RDD of lab result
    * @return RDD of feature tuples
    */


  def constructMoCARDDFeatureTuple(moCARDD: RDD[MOCA]): RDD[FeatureTuple] = {
    val featureTuples = moCARDD.map(s=>((s.patientID, "MoCA"), s.Score.toDouble))
    featureTuples
  }

  def constructUpdrsIRDDFeatureTuple(updrsIRDD: RDD[UPDRS]): RDD[FeatureTuple] = {
    val featureTuples = updrsIRDD.map(s=>((s.patientID, "UPDRSI"), s.Score.toDouble))
    featureTuples
  }

  def constructUpdrsIIRDDFeatureTuple(updrsIIRDD: RDD[UPDRS]): RDD[FeatureTuple] = {
    val featureTuples = updrsIIRDD.map(s=>((s.patientID, "UPDRSII"), s.Score.toDouble))
    featureTuples
  }

  def constructUpdrsIIIRDDFeatureTuple(updrsIIIRDD: RDD[UPDRS]): RDD[FeatureTuple] = {
    val featureTuples = updrsIIIRDD.map(s=>((s.patientID, "UPDRSIII"), s.Score.toDouble))
    featureTuples
  }


  def constructBioChemRDDFeatureTuple(bioChemRDD: RDD[BIOCHEM]): RDD[FeatureTuple] = {
    val featureTuples = bioChemRDD.map(s=>
      if (s.testName == "ptau" && s.value == "<8") ((s.patientID, s.testName), 8.0)
      else if (s.testName == "ttau" && s.value == "<80") ((s.patientID, s.testName), 80.0)
      else if (s.testName == "abeta 1-42" && s.value== "<200") ((s.patientID, s.testName), 200.0)
      else if (s.testName == "abeta 1-42" && s.value== ">1700") ((s.patientID, s.testName), 1700.0)


      else if (s.testName == "apoe genotype" && s.value== "e3/e3") ((s.patientID, s.testName), 6.0)
      else if (s.testName == "apoe genotype" && s.value== "e2/e4") ((s.patientID, s.testName), 6.0)
      else if (s.testName == "apoe genotype" && s.value== "e3/e2") ((s.patientID, s.testName), 5.0)
      else if (s.testName == "apoe genotype" && s.value== "e4/e3") ((s.patientID, s.testName), 7.0)
      else if (s.testName == "apoe genotype" && s.value== "e4/e4") ((s.patientID, s.testName), 8.0)
      else if (s.testName == "apoe genotype" && s.value== "e2/e2") ((s.patientID, s.testName), 4.0)

      else if (s.testName == "rs76763715_gba_p.n370s" && s.value== "C/T") ((s.patientID, s.testName), 0.0)
      else if (s.testName == "rs76763715_gba_p.n370s" && s.value== "T/T") ((s.patientID, s.testName), 1.0)

      else if (s.testName == "rs34637584_lrrk2_p.g2019s" && s.value== "A/G") ((s.patientID, s.testName), 0.0)
      else if (s.testName == "rs34637584_lrrk2_p.g2019s" && s.value== "G/G") ((s.patientID, s.testName), 1.0)

      else if (s.testName == "rs17649553" && s.value== "C/C") ((s.patientID, s.testName), 0.0)
      else if (s.testName == "rs17649553" && s.value== "C/T") ((s.patientID, s.testName), 1.0)
      else if (s.testName == "rs17649553" && s.value== "T/T") ((s.patientID, s.testName), 2.0)

      else if (s.testName == "rs3910105" && s.value== "C/C") ((s.patientID, s.testName), 0.0)
      else if (s.testName == "rs3910105" && s.value== "C/T") ((s.patientID, s.testName), 1.0)
      else if (s.testName == "rs3910105" && s.value== "T/T") ((s.patientID, s.testName), 2.0)

      else if (s.testName == "rs356181" && s.value== "C/C") ((s.patientID, s.testName), 0.0)
      else if (s.testName == "rs356181" && s.value== "C/T") ((s.patientID, s.testName), 1.0)
      else if (s.testName == "rs356181" && s.value== "T/T") ((s.patientID, s.testName), 2.0)

      else ((s.patientID, s.testName), s.value.toString.toDouble)
    )

/*    val temp = featureTuples.filter(s=>s._1._2 == "ptau"||s._1._2 == "ttau"||s._1._2 == "abeta 1-42").
      map(s=>(s._1._1, (s._1._2, s._2))).
      groupByKey()*/




    val bioChemCount = featureTuples.map(s=>(s._1,1.0)).reduceByKey(_+_)
    val bioChemSum = featureTuples.reduceByKey(_+_)
    val bioChemfeatureTuples = bioChemCount.join(bioChemSum).map(s=>(s._1, s._2._2.toString.toDouble/s._2._1))

    // why some of the baseline was took several years apart???

    bioChemfeatureTuples
  }

  // ABeta 1-42	                <200 >1700
  // pTau	                      <8
  // tTau	                      <80
  // APOE GENOTYPE              e3/e3, e2/e4, e3/e2, e4/e3, e4/e4, e2/e2)
  // rs76763715_GBA_p.N370S     C/T, T/T
  // rs35801418_LRRK2_p.Y1699C  A/A                 //not necessary
  // rs35870237_LRRK2_p.I2020T  T/T                 //not necessary
  // rs34637584_LRRK2_p.G2019S  A/G, G/G
  // rs34995376_LRRK2_p.R1441H  G/G                 //not necessary
  // rs17649553                 C/T, T/T, C/C
  // rs3910105                  C/T, T/T, C/C
  // rs356181                   C/T, T/T, C/C
  // SNCA_multiplication        CopyNumberChange, NotAssessed, NormalCopyNumber)  //Not necessary

// could StandardScaler accept numeric and categorical features? feature a: 2.3, feature b, [1, 0, 0]





  /**
   * Given a feature tuples RDD, construct features in vector
   * format for each patient. feature name should be mapped
   * to some index and convert to sparse feature format.
   * @param sc SparkContext to run
   * @param feature RDD of input feature tuples
   * @return
   */
  def construct(sc: SparkContext, feature: RDD[FeatureTuple], phenotypeLabel:RDD[(String, Int)]): RDD[(String, Vector)] = {

    /** save for later usage */
    feature.cache()
    //val scFeature = sc.broadcast(feature) //So we need this?

    /** create a feature name to id map*/

    val featureID = feature.map(s=>s._1._2).distinct().collect().zipWithIndex.toMap //(feature name -> feature ID)
    val numFeature = featureID.keys.size

    println(numFeature)

    /** transform input feature */

    val result = feature.map(s=>(s._1._1, (s._1._2, s._2))).groupByKey().map{s=>
      val indexedFeatures = s._2.toList.map(s=>(featureID(s._1),s._2))
      val featureVector = Vectors.sparse(numFeature, indexedFeatures)
      val res = (s._1, featureVector)
      res
    }

    result

  }



  /** save features to svmlight format for SKlearn
    * @param sc SparkContext to run
    * @param feature RDD of input feature tuples
    * @return
    * */

  def saveFeatures(sc: SparkContext, feature: RDD[FeatureTuple], phenotypeLabel:RDD[(String, Int)], updrs: Int): Unit = {

    /** save for later usage */
    feature.cache()

    // Generate training data for SKlearn
    val featureID = feature.map(s=>s._1._2).distinct().collect().zipWithIndex.toMap //(feature name -> feature ID)

    println("Generating training data for SKlearn")

    import java.io._

    var pw = new PrintWriter(new File("output/withoutUPDRS.train"))

    if (updrs == 0){
      pw = new PrintWriter(new File("output/withoutUPDRS.train"))
    }
    else{
      pw = new PrintWriter(new File("output/withUPDRS.train"))
    }


    val temp = feature.map(s=>(s._1._1, (s._1._2, s._2))).groupByKey().map{s=>
      val indexedFeatures = s._2.toList.map(s=>(featureID(s._1),s._2))
      (s._1, indexedFeatures)
    }

    val temp2 = temp.join(phenotypeLabel).map(s=>(s._2._2.toString, s._2._1)).collect()

    for (patientFeature <- temp2){
      pw.printf(patientFeature._1)
      for (feat <- patientFeature._2.sortBy(s=>s._1)){
        pw.print(" ")
        pw.print(feat._1)
        pw.print(":")
        pw.print(feat._2)
      }
      pw.printf("\n")
    }

    pw.close()

    println("Done")

  }
}


