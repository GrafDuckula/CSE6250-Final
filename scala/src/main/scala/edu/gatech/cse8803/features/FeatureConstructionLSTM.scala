/**
 * @author Hang Su
 */
package edu.gatech.cse8803.features

import java.io.{File, PrintWriter}

import edu.gatech.cse8803.model._
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._


object FeatureConstructionLSTM {

  /**
   * ((patient-id, feature-name), feature-value)
   */
  type FeatureTuple = ((String, String, String), Double)

  val timestampSC = List("SC", "V01", "V02", "V03", "V04", "V05", "V06", "V07", "V08", "V09", "V10", "V11", "V12", "V13", "V14", "V15")
  val timestampBL = List("BL", "V01", "V02", "V03", "V04", "V05", "V06", "V07", "V08", "V09", "V10", "V11", "V12", "V13", "V14", "V15")

  /**
   * Aggregate feature tuples from diagnostic with COUNT aggregation,
   * @param socEcoRDD RDD of education years
   * @return RDD of feature tuples
   */
  def constructSocEcoFeatureTuple(socEcoRDD: RDD[EDU]): RDD[FeatureTuple] = {
    val featureTuples = socEcoRDD.map(s=>((s.patientID, "SC", "eduYears"),s.years.toDouble))
    featureTuples
  }

  /**
   * Aggregate feature tuples from medication with COUNT aggregation,
   * @param scDemogrRDD RDD of medication
   * @return RDD of feature tuples
   */
  def constructScDemogrFeatureTuple(scDemogrRDD: RDD[DEMOGR]): RDD[FeatureTuple] = {
    val featureTuples_G = scDemogrRDD.map(s=>((s.patientID, "SC", "gender"), s.gender.toDouble))
    val featureTuples_A = scDemogrRDD.map(s=>((s.patientID, "SC", "age"), s.age))
    featureTuples_G.union(featureTuples_A)
  }

  def constructFamHistFeatureTuple(famHistRDD: RDD[HIST]): RDD[FeatureTuple] = {
    val featureTuples = famHistRDD.
      filter(s=>s.eventID=="SC").map(s=>((s.patientID, "SC", "familyHistory"),s.Score))
    featureTuples
  }



  /**
   * Aggregate feature tuples from lab result, using AVERAGE aggregation
   * @param datScanRDD RDD of lab result
   * @return RDD of feature tuples
   */
  def constructDatScanRDDFeatureTuple(datScanRDD: RDD[DATSCAN]): RDD[FeatureTuple] = {
    val datScanRDDSc = datScanRDD.filter(s=>timestampSC.contains(s.eventID))
    val featureTuples_C = datScanRDDSc.map(s=>((s.patientID, s.eventID, "CAUDATE"), s.CAUDATE))
    val featureTuples_P = datScanRDDSc.map(s=>((s.patientID, s.eventID, "PUTAMEN"), s.PUTAMEN))
    val featureTuples_R = datScanRDDSc.map(s=>((s.patientID, s.eventID, "CAUDATE_PUTAMEN_ratio"), s.cpRatio))
    val featureTuples_CA = datScanRDDSc.map(s=>((s.patientID, s.eventID, "CAUDATE_Asym"), s.caudateAsym))
    val featureTuples_PA = datScanRDDSc.map(s=>((s.patientID, s.eventID, "PUTAMEN_Asym"), s.putamenAsym))
    featureTuples_C.union(featureTuples_P).union(featureTuples_R).union(featureTuples_CA).union(featureTuples_PA)
  }



  def constructQuipRDDFeatureTuple(quipRDD: RDD[QUIP]): RDD[FeatureTuple] = {
    val featureTuples = quipRDD.filter(s=>timestampBL.contains(s.eventID)).
      map(s=>((s.patientID, s.eventID, "QUIP"), s.Score.toDouble))
    featureTuples
  }

  def constructHvltRDDFeatureTuple(hvltRDD: RDD[HVLT]): RDD[FeatureTuple] = {
    val hvltRDDBL = hvltRDD.filter(s=>timestampBL.contains(s.eventID))
    val featureTuples_I = hvltRDDBL.map(s=>((s.patientID, s.eventID, "HVLT_immediateScore"), s.immediateScore.toDouble))
    val featureTuples_D = hvltRDDBL.map(s=>((s.patientID, s.eventID, "HVLT_discrimScore"), s.discrimScore.toDouble))
    val featureTuples_R = hvltRDDBL.map(s=>((s.patientID, s.eventID, "HVLT_retentScore"), s.retentScore))
    featureTuples_I.union(featureTuples_D).union(featureTuples_R)
  }


  def constructBLineRDDFeatureTuple(bLineRDD: RDD[BLINE]): RDD[FeatureTuple] = {
    val featureTuples = bLineRDD.filter(s=>timestampBL.contains(s.eventID)).
      map(s=>((s.patientID, s.eventID, "Benton_Line_Orientation"), s.Score.toDouble))
    featureTuples
  }

  def constructSemaFlutRDDFeatureTuple(semaFlutRDD: RDD[SFT]): RDD[FeatureTuple] = {
    val semaFlutRDDBL = semaFlutRDD.filter(s=>timestampBL.contains(s.eventID))
    val featureTuples_S = semaFlutRDDBL.map(s=>((s.patientID, s.eventID, "SemaFlut_sumScore"), s.sumScore.toDouble))
    val featureTuples_DT = semaFlutRDDBL.map(s=>((s.patientID, s.eventID, "SemaFlut_derivedScaled"), s.derivedSemScaled.toDouble))
    val featureTuples_DS = semaFlutRDDBL.map(s=>((s.patientID, s.eventID, "SemaFlut_derivedT"), s.derivedSemT.toDouble))
    featureTuples_S.union(featureTuples_DT).union(featureTuples_DS)
  }

  def constructLnSeqRDDFeatureTuple(lnSeqRDD: RDD[LNSEQ]): RDD[FeatureTuple] = {
    val featureTuples_S = lnSeqRDD.filter(s=>timestampBL.contains(s.eventID)).
      map(s=>((s.patientID, s.eventID, "LnSeq_Score"), s.Score.toDouble))
    val featureTuples_DS = lnSeqRDD.filter(s=>timestampBL.contains(s.eventID)).
      map(s=>((s.patientID, s.eventID, "LnSeq_derivedScore"), s.derivedScore.toDouble))
    featureTuples_S.union(featureTuples_DS)
  }

  def constructSdModRDDFeatureTuple(sdModRDD: RDD[SDMOD]): RDD[FeatureTuple] = {
    val featureTuples_S = sdModRDD.filter(s=>timestampBL.contains(s.eventID)).
      map(s=>((s.patientID, s.eventID, "SdMod_Score"), s.Score.toDouble))
    val featureTuples_DT = sdModRDD.filter(s=>timestampBL.contains(s.eventID)).
      map(s=>((s.patientID, s.eventID, "SdMod_derivedTScore"), s.derivedTScore.toDouble))
    featureTuples_S.union(featureTuples_DT)
  }

  def constructEssRDDFeatureTuple(essRDD: RDD[ESS]): RDD[FeatureTuple] = {
    val featureTuples = essRDD.filter(s=>timestampBL.contains(s.eventID)).
      map(s=>((s.patientID, s.eventID, "ESS"), s.Score.toDouble))
    featureTuples
  }

  def constructStaiRDDFeatureTuple(staiRDD: RDD[STAI]): RDD[FeatureTuple] = {
    val staiRDDBL = staiRDD.filter(s=>timestampBL.contains(s.eventID))
    val featureTuples_S = staiRDDBL.map(s=>((s.patientID, s.eventID, "STAI_Score"), s.Score.toDouble))
    val featureTuples_SS = staiRDDBL.map(s=>((s.patientID, s.eventID, "STAI_stateScore"), s.stateScore.toDouble))
    val featureTuples_TS = staiRDDBL.map(s=>((s.patientID, s.eventID, "STAI_traitScore"), s.traitScore.toDouble))
    featureTuples_S.union(featureTuples_SS).union(featureTuples_TS)
  }

  def constructGdsRDDFeatureTuple(gdsRDD: RDD[GDS]): RDD[FeatureTuple] = {
    val featureTuples = gdsRDD.filter(s=>timestampBL.contains(s.eventID)).
      map(s => ((s.patientID, s.eventID, "Gds_Score"), s.Score.toDouble))
    featureTuples
  }

  def constructScopaRDDFeatureTuple(scopaRDD: RDD[SCOPA]): RDD[FeatureTuple] = {
    val featureTuples = scopaRDD.filter(s=>timestampBL.contains(s.eventID)).
      map(s=>((s.patientID, s.eventID, "SCOPA"), s.Score.toDouble))
    featureTuples
  }



  /**
    * Aggregate feature tuples from lab result, using AVERAGE aggregation
    * @param upsitRDD RDD of lab result
    * @return RDD of feature tuples
    */
  // PD and HC patients only have BL event, other patients might have V0X event.
  def constructUpsitRDDFeatureTuple(upsitRDD: RDD[UPSIT]): RDD[FeatureTuple] = {
    val featureTuples = upsitRDD.filter(s=>s.eventID=="BL").map(s=>((s.patientID, s.eventID, "UPSIT"), s.Score.toDouble))
    featureTuples
  }


  /**
    * Aggregate feature tuples from lab result, using AVERAGE aggregation
    * @param remRDD RDD of lab result
    * @return RDD of feature tuples
    */

  def constructRemRDDFeatureTuple(remRDD: RDD[REM]): RDD[FeatureTuple] = {
    val featureTuples = remRDD.filter(s=>timestampBL.contains(s.eventID)).
      map(s=>((s.patientID, s.eventID, "REM"), s.Score.toDouble))
    featureTuples
  }

  /**
    * Aggregate feature tuples from lab result, using AVERAGE aggregation
    * @param moCARDD RDD of lab result
    * @return RDD of feature tuples
    */

  // PD and HC patients only have SC event
  def constructMoCARDDFeatureTuple(moCARDD: RDD[MOCA]): RDD[FeatureTuple] = {
    val featureTuples = moCARDD.filter(s=>timestampSC.contains(s.eventID)).
      map(s=>((s.patientID, s.eventID, "MoCA"), s.Score.toDouble))
    featureTuples
  }

  // Used BL instead of SC
  def constructUpdrsIRDDFeatureTuple(updrsIRDD: RDD[UPDRS]): RDD[FeatureTuple] = {
    val featureTuples = updrsIRDD.filter(s=>timestampBL.contains(s.eventID)).
      map(s=>((s.patientID, s.eventID, "UPDRSI"), s.Score.toDouble))
    featureTuples
  }

  def constructUpdrsIIRDDFeatureTuple(updrsIIRDD: RDD[UPDRS]): RDD[FeatureTuple] = {
    val featureTuples = updrsIIRDD.filter(s=>timestampBL.contains(s.eventID)).
      map(s=>((s.patientID, s.eventID, "UPDRSII"), s.Score.toDouble))
    featureTuples
  }

  def constructUpdrsIIIRDDFeatureTuple(updrsIIIRDD: RDD[UPDRS]): RDD[FeatureTuple] = {
    val featureTuples = updrsIIIRDD.filter(s=>timestampBL.contains(s.eventID)).
      map(s=>((s.patientID, s.eventID, "UPDRSIII"), s.Score.toDouble))
    featureTuples
  }

  // All SC is for DNA
  def constructBioChemRDDFeatureTuple(bioChemRDD: RDD[BIOCHEM]): RDD[FeatureTuple] = {

/*    val featureTuples = bioChemRDD.map(s=>
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
    )*/

    val featureTuples = bioChemRDD.filter(s=>timestampBL.contains(s.eventID)||s.eventID=="SC").flatMap{ s =>
      if (s.testName == "ptau" && s.value == "<8") Seq(((s.patientID, s.eventID, s.testName), 8.0))
      else if (s.testName == "ttau" && s.value == "<80") Seq(((s.patientID, s.eventID, s.testName), 80.0))
      else if (s.testName == "abeta 1-42" && s.value== "<200") Seq(((s.patientID, s.eventID, s.testName), 200.0))
      else if (s.testName == "abeta 1-42" && s.value== ">1700") Seq(((s.patientID, s.eventID, s.testName), 1700.0))

      else if (s.testName == "apoe genotype" && s.value== "e3/e3") Seq(((s.patientID, s.eventID, "apoe_e2"), 0.0), ((s.patientID, s.eventID, "apoe_e3"), 2.0), ((s.patientID, s.eventID, "apoe_e4"), 0.0))
      else if (s.testName == "apoe genotype" && s.value== "e2/e4") Seq(((s.patientID, s.eventID, "apoe_e2"), 1.0), ((s.patientID, s.eventID, "apoe_e3"), 0.0), ((s.patientID, s.eventID, "apoe_e4"), 1.0))
      else if (s.testName == "apoe genotype" && s.value== "e3/e2") Seq(((s.patientID, s.eventID, "apoe_e2"), 1.0), ((s.patientID, s.eventID, "apoe_e3"), 1.0), ((s.patientID, s.eventID, "apoe_e4"), 0.0))
      else if (s.testName == "apoe genotype" && s.value== "e4/e3") Seq(((s.patientID, s.eventID, "apoe_e2"), 0.0), ((s.patientID, s.eventID, "apoe_e3"), 1.0), ((s.patientID, s.eventID, "apoe_e4"), 1.0))
      else if (s.testName == "apoe genotype" && s.value== "e4/e4") Seq(((s.patientID, s.eventID, "apoe_e2"), 0.0), ((s.patientID, s.eventID, "apoe_e3"), 0.0), ((s.patientID, s.eventID, "apoe_e4"), 2.0))
      else if (s.testName == "apoe genotype" && s.value== "e2/e2") Seq(((s.patientID, s.eventID, "apoe_e2"), 2.0), ((s.patientID, s.eventID, "apoe_e3"), 0.0), ((s.patientID, s.eventID, "apoe_e4"), 0.0))


      else if (s.testName == "rs76763715_gba_p.n370s" && s.value== "C/T") Seq(((s.patientID, s.eventID, "gba_C"), 1.0), ((s.patientID, s.eventID, "gba_T"), 1.0))
      else if (s.testName == "rs76763715_gba_p.n370s" && s.value== "T/T") Seq(((s.patientID, s.eventID, "gba_C"), 0.0), ((s.patientID, s.eventID, "gba_T"), 2.0))


      else if (s.testName == "rs34637584_lrrk2_p.g2019s" && s.value== "A/G") Seq(((s.patientID, s.eventID, "lrrk2_A"), 1.0), ((s.patientID, s.eventID, "lrrk2_G"), 1.0))
      else if (s.testName == "rs34637584_lrrk2_p.g2019s" && s.value== "G/G") Seq(((s.patientID, s.eventID, "lrrk2_A"), 0.0), ((s.patientID, s.eventID, "lrrk2_G"), 2.0))


      else if (s.testName == "rs17649553" && s.value== "C/C") Seq(((s.patientID, s.eventID, "rs17649553_C"), 2.0), ((s.patientID, s.eventID, "rs17649553_T"), 0.0))
      else if (s.testName == "rs17649553" && s.value== "C/T") Seq(((s.patientID, s.eventID, "rs17649553_C"), 1.0), ((s.patientID, s.eventID, "rs17649553_T"), 1.0))
      else if (s.testName == "rs17649553" && s.value== "T/T") Seq(((s.patientID, s.eventID, "rs17649553_C"), 0.0), ((s.patientID, s.eventID, "rs17649553_T"), 2.0))


      else if (s.testName == "rs3910105" && s.value== "C/C") Seq(((s.patientID, s.eventID, "rs3910105_C"), 2.0), ((s.patientID, s.eventID, "rs3910105_T"), 0.0))
      else if (s.testName == "rs3910105" && s.value== "C/T") Seq(((s.patientID, s.eventID, "rs3910105_C"), 1.0), ((s.patientID, s.eventID, "rs3910105_T"), 1.0))
      else if (s.testName == "rs3910105" && s.value== "T/T") Seq(((s.patientID, s.eventID, "rs3910105_C"), 0.0), ((s.patientID, s.eventID, "rs3910105_T"), 2.0))


      else if (s.testName == "rs356181" && s.value== "C/C") Seq(((s.patientID, s.eventID, "rs356181_C"), 2.0), ((s.patientID, s.eventID, "rs356181_T"), 0.0))
      else if (s.testName == "rs356181" && s.value== "C/T") Seq(((s.patientID, s.eventID, "rs356181_C"), 1.0), ((s.patientID, s.eventID, "rs356181_T"), 1.0))
      else if (s.testName == "rs356181" && s.value== "T/T") Seq(((s.patientID, s.eventID, "rs356181_C"), 0.0), ((s.patientID, s.eventID, "rs356181_T"), 2.0))


     /* Maybe not important*/

      else if (s.testName == "rs35801418_lrrk2_p.y1699c" && s.value== "A/A") Seq(((s.patientID, s.eventID, "Y1699C_A"), 2.0)) //not necessary
      else if (s.testName == "rs35870237_lrrk2_p.i2020t" && s.value== "T/T") Seq(((s.patientID, s.eventID, "I2020T_T"), 2.0)) //not necessary
      else if (s.testName == "rs34995376_lrrk2_p.r1441h" && s.value== "G/G") Seq(((s.patientID, s.eventID, "R1441H_G"), 2.0)) //not necessary

      else if (s.testName == "snca_multiplication") Seq() // CopyNumberChange, NotAssessed, NormalCopyNumber)  //Not necessary
      else if (s.testName == "score" && s.testType == "dna" && (s.value == "8.21E-05" || s.value == "4.64E-05")) Seq() // I am not sure what DNA score means
      else if (s.testType == "rna" && s.unit== "SD") Seq()
      else if (s.testName == "pd2" && s.unit== "Stdev") Seq()
      else if (s.testName == "PD2 Peptoid" && s.value.toString.toDouble < 0) Seq(((s.patientID, s.eventID, "PD2 Peptoid"), 0.0))


      else if (s.testName == "ldl" && s.value== "below detection limit") Seq(((s.patientID, s.eventID, "LDL"), 12.0))
      else if (s.testName == "triglycerides" && s.value== "below detection limit") Seq(((s.patientID, s.eventID, "triglycerides"), 20.0))
      else if (s.testName == "total cholesterol" && s.value== "below detection limit") Seq(((s.patientID, s.eventID, "total cholesterol"), 85.0))

      else if (s.testName == "csf hemoglobin" && s.value== "below detection limit") Seq(((s.patientID, s.eventID, "csf hemoglobin"), 30.0))
      else if (s.testName == "csf hemoglobin" && s.value== "below") Seq(((s.patientID, s.eventID, "csf hemoglobin"), 30.0))
      else if (s.testName == "csf hemoglobin" && s.value== "above") Seq(((s.patientID, s.eventID, "csf hemoglobin"), 12500.0))
      else if (s.testName == "csf hemoglobin" && s.value== ">12500 ng/ml") Seq(((s.patientID, s.eventID, "csf hemoglobin"), 12500.0))
      else if (s.testName == "csf hemoglobin" && s.value== ">12500ng/ml") Seq(((s.patientID, s.eventID, "csf hemoglobin"), 12500.0))
      else if (s.testName == "csf hemoglobin" && s.value== ">20") Seq(((s.patientID, s.eventID, "csf hemoglobin"), 20.0)) //???


      // CT to TC mutation
      else if (s.testName == "rs10797576" && s.value== "C/C") Seq(((s.patientID, s.eventID, "rs10797576_C"), 2.0), ((s.patientID, s.eventID, "rs10797576_T"), 0.0))
      else if (s.testName == "rs10797576" && s.value== "C/T") Seq(((s.patientID, s.eventID, "rs10797576_C"), 1.0), ((s.patientID, s.eventID, "rs10797576_T"), 1.0))
      else if (s.testName == "rs10797576" && s.value== "T/T") Seq(((s.patientID, s.eventID, "rs10797576_C"), 0.0), ((s.patientID, s.eventID, "rs10797576_T"), 2.0))

      else if (s.testName == "rs11158026" && s.value== "C/C") Seq(((s.patientID, s.eventID, "rs11158026_C"), 2.0), ((s.patientID, s.eventID, "rs11158026_T"), 0.0))
      else if (s.testName == "rs11158026" && s.value== "C/T") Seq(((s.patientID, s.eventID, "rs11158026_C"), 1.0), ((s.patientID, s.eventID, "rs11158026_T"), 1.0))
      else if (s.testName == "rs11158026" && s.value== "T/T") Seq(((s.patientID, s.eventID, "rs11158026_C"), 0.0), ((s.patientID, s.eventID, "rs11158026_T"), 2.0))

      else if (s.testName == "rs115462410" && s.value== "C/C") Seq(((s.patientID, s.eventID, "rs115462410_C"), 2.0), ((s.patientID, s.eventID, "rs115462410_T"), 0.0))
      else if (s.testName == "rs115462410" && s.value== "C/T") Seq(((s.patientID, s.eventID, "rs115462410_C"), 1.0), ((s.patientID, s.eventID, "rs115462410_T"), 1.0))
      else if (s.testName == "rs115462410" && s.value== "T/T") Seq(((s.patientID, s.eventID, "rs115462410_C"), 0.0), ((s.patientID, s.eventID, "rs115462410_T"), 2.0))

      else if (s.testName == "rs199347" && s.value== "C/C") Seq(((s.patientID, s.eventID, "rs199347_C"), 2.0), ((s.patientID, s.eventID, "rs199347_T"), 0.0))
      else if (s.testName == "rs199347" && s.value== "C/T") Seq(((s.patientID, s.eventID, "rs199347_C"), 1.0), ((s.patientID, s.eventID, "rs199347_T"), 1.0))
      else if (s.testName == "rs199347" && s.value== "T/T") Seq(((s.patientID, s.eventID, "rs199347_C"), 0.0), ((s.patientID, s.eventID, "rs199347_T"), 2.0))

      else if (s.testName == "rs329648" && s.value== "C/C") Seq(((s.patientID, s.eventID, "rs329648_C"), 2.0), ((s.patientID, s.eventID, "rs329648_T"), 0.0))
      else if (s.testName == "rs329648" && s.value== "C/T") Seq(((s.patientID, s.eventID, "rs329648_C"), 1.0), ((s.patientID, s.eventID, "rs329648_T"), 1.0))
      else if (s.testName == "rs329648" && s.value== "T/T") Seq(((s.patientID, s.eventID, "rs329648_C"), 0.0), ((s.patientID, s.eventID, "rs329648_T"), 2.0))

      else if (s.testName == "rs6430538" && s.value== "C/C") Seq(((s.patientID, s.eventID, "rs6430538_C"), 2.0), ((s.patientID, s.eventID, "rs6430538_T"), 0.0))
      else if (s.testName == "rs6430538" && s.value== "C/T") Seq(((s.patientID, s.eventID, "rs6430538_C"), 1.0), ((s.patientID, s.eventID, "rs6430538_T"), 1.0))
      else if (s.testName == "rs6430538" && s.value== "T/T") Seq(((s.patientID, s.eventID, "rs6430538_C"), 0.0), ((s.patientID, s.eventID, "rs6430538_T"), 2.0))

      else if (s.testName == "rs6812193" && s.value== "C/C") Seq(((s.patientID, s.eventID, "rs6812193_C"), 2.0), ((s.patientID, s.eventID, "rs6812193_T"), 0.0))
      else if (s.testName == "rs6812193" && s.value== "C/T") Seq(((s.patientID, s.eventID, "rs6812193_C"), 1.0), ((s.patientID, s.eventID, "rs6812193_T"), 1.0))
      else if (s.testName == "rs6812193" && s.value== "T/T") Seq(((s.patientID, s.eventID, "rs6812193_C"), 0.0), ((s.patientID, s.eventID, "rs6812193_T"), 2.0))

      else if (s.testName == "rs76904798" && s.value== "C/C") Seq(((s.patientID, s.eventID, "rs76904798_C"), 2.0), ((s.patientID, s.eventID, "rs76904798_T"), 0.0))
      else if (s.testName == "rs76904798" && s.value== "C/T") Seq(((s.patientID, s.eventID, "rs76904798_C"), 1.0), ((s.patientID, s.eventID, "rs76904798_T"), 1.0))
      else if (s.testName == "rs76904798" && s.value== "T/T") Seq(((s.patientID, s.eventID, "rs76904798_C"), 0.0), ((s.patientID, s.eventID, "rs76904798_T"), 2.0))

      else if (s.testName == "rs823118" && s.value== "C/C") Seq(((s.patientID, s.eventID, "rs823118_C"), 2.0), ((s.patientID, s.eventID, "rs823118_T"), 0.0))
      else if (s.testName == "rs823118" && s.value== "C/T") Seq(((s.patientID, s.eventID, "rs823118_C"), 1.0), ((s.patientID, s.eventID, "rs823118_T"), 1.0))
      else if (s.testName == "rs823118" && s.value== "T/T") Seq(((s.patientID, s.eventID, "rs823118_C"), 0.0), ((s.patientID, s.eventID, "rs823118_T"), 2.0))


      // GT or TG mutation
      else if (s.testName == "rs1955337" && s.value== "G/G") Seq(((s.patientID, s.eventID, "rs1955337_G"), 2.0), ((s.patientID, s.eventID, "rs1955337_T"), 0.0))
      else if (s.testName == "rs1955337" && s.value== "G/T") Seq(((s.patientID, s.eventID, "rs1955337_G"), 1.0), ((s.patientID, s.eventID, "rs1955337_T"), 1.0))
      else if (s.testName == "rs1955337" && s.value== "T/T") Seq(((s.patientID, s.eventID, "rs1955337_G"), 0.0), ((s.patientID, s.eventID, "rs1955337_T"), 2.0))

      else if (s.testName == "rs34884217" && s.value== "G/G") Seq(((s.patientID, s.eventID, "rs34884217_G"), 2.0), ((s.patientID, s.eventID, "rs34884217_T"), 0.0))
      else if (s.testName == "rs34884217" && s.value== "G/T") Seq(((s.patientID, s.eventID, "rs34884217_G"), 1.0), ((s.patientID, s.eventID, "rs34884217_T"), 1.0))
      else if (s.testName == "rs34884217" && s.value== "T/T") Seq(((s.patientID, s.eventID, "rs34884217_G"), 0.0), ((s.patientID, s.eventID, "rs34884217_T"), 2.0))

      // AG or GA Mutation
      else if (s.testName == "rs11060180" && s.value== "G/G") Seq(((s.patientID, s.eventID, "rs11060180_G"), 2.0), ((s.patientID, s.eventID, "rs11060180_A"), 0.0))
      else if (s.testName == "rs11060180" && s.value== "A/G") Seq(((s.patientID, s.eventID, "rs11060180_G"), 1.0), ((s.patientID, s.eventID, "rs11060180_A"), 1.0))
      else if (s.testName == "rs11060180" && s.value== "A/A") Seq(((s.patientID, s.eventID, "rs11060180_G"), 0.0), ((s.patientID, s.eventID, "rs11060180_A"), 2.0))

      else if (s.testName == "rs11868035" && s.value== "G/G") Seq(((s.patientID, s.eventID, "rs11868035_G"), 2.0), ((s.patientID, s.eventID, "rs11868035_A"), 0.0))
      else if (s.testName == "rs11868035" && s.value== "A/G") Seq(((s.patientID, s.eventID, "rs11868035_G"), 1.0), ((s.patientID, s.eventID, "rs11868035_A"), 1.0))
      else if (s.testName == "rs11868035" && s.value== "A/A") Seq(((s.patientID, s.eventID, "rs11868035_G"), 0.0), ((s.patientID, s.eventID, "rs11868035_A"), 2.0))

      else if (s.testName == "rs12456492" && s.value== "G/G") Seq(((s.patientID, s.eventID, "rs12456492_G"), 2.0), ((s.patientID, s.eventID, "rs12456492_A"), 0.0))
      else if (s.testName == "rs12456492" && s.value== "A/G") Seq(((s.patientID, s.eventID, "rs12456492_G"), 1.0), ((s.patientID, s.eventID, "rs12456492_A"), 1.0))
      else if (s.testName == "rs12456492" && s.value== "A/A") Seq(((s.patientID, s.eventID, "rs12456492_G"), 0.0), ((s.patientID, s.eventID, "rs12456492_A"), 2.0))

      else if (s.testName == "rs12637471" && s.value== "G/G") Seq(((s.patientID, s.eventID, "rs12637471_G"), 2.0), ((s.patientID, s.eventID, "rs12637471_A"), 0.0))
      else if (s.testName == "rs12637471" && s.value== "A/G") Seq(((s.patientID, s.eventID, "rs12637471_G"), 1.0), ((s.patientID, s.eventID, "rs12637471_A"), 1.0))
      else if (s.testName == "rs12637471" && s.value== "A/A") Seq(((s.patientID, s.eventID, "rs12637471_G"), 0.0), ((s.patientID, s.eventID, "rs12637471_A"), 2.0))

      else if (s.testName == "rs14235" && s.value== "G/G") Seq(((s.patientID, s.eventID, "rs14235_G"), 2.0), ((s.patientID, s.eventID, "rs14235_A"), 0.0))
      else if (s.testName == "rs14235" && s.value== "A/G") Seq(((s.patientID, s.eventID, "rs14235_G"), 1.0), ((s.patientID, s.eventID, "rs14235_A"), 1.0))
      else if (s.testName == "rs14235" && s.value== "A/A") Seq(((s.patientID, s.eventID, "rs14235_G"), 0.0), ((s.patientID, s.eventID, "rs14235_A"), 2.0))



      else if (s.testName == "rs2414739" && s.value== "G/G") Seq(((s.patientID, s.eventID, "rs2414739_G"), 2.0), ((s.patientID, s.eventID, "rs2414739_A"), 0.0))
      else if (s.testName == "rs2414739" && s.value== "A/G") Seq(((s.patientID, s.eventID, "rs2414739_G"), 1.0), ((s.patientID, s.eventID, "rs2414739_A"), 1.0))
      else if (s.testName == "rs2414739" && s.value== "A/A") Seq(((s.patientID, s.eventID, "rs2414739_G"), 0.0), ((s.patientID, s.eventID, "rs2414739_A"), 2.0))

      else if (s.testName == "rs34311866" && s.value== "G/G") Seq(((s.patientID, s.eventID, "rs34311866_G"), 2.0), ((s.patientID, s.eventID, "rs34311866_A"), 0.0))
      else if (s.testName == "rs34311866" && s.value== "A/G") Seq(((s.patientID, s.eventID, "rs34311866_G"), 1.0), ((s.patientID, s.eventID, "rs34311866_A"), 1.0))
      else if (s.testName == "rs34311866" && s.value== "A/A") Seq(((s.patientID, s.eventID, "rs34311866_G"), 0.0), ((s.patientID, s.eventID, "rs34311866_A"), 2.0))

      else if (s.testName == "rs55785911" && s.value== "G/G") Seq(((s.patientID, s.eventID, "rs55785911_G"), 2.0), ((s.patientID, s.eventID, "rs55785911_A"), 0.0))
      else if (s.testName == "rs55785911" && s.value== "A/G") Seq(((s.patientID, s.eventID, "rs55785911_G"), 1.0), ((s.patientID, s.eventID, "rs55785911_A"), 1.0))
      else if (s.testName == "rs55785911" && s.value== "A/A") Seq(((s.patientID, s.eventID, "rs55785911_G"), 0.0), ((s.patientID, s.eventID, "rs55785911_A"), 2.0))

      else if (s.testName == "rs591323" && s.value== "G/G") Seq(((s.patientID, s.eventID, "rs591323_G"), 2.0), ((s.patientID, s.eventID, "rs591323_A"), 0.0))
      else if (s.testName == "rs591323" && s.value== "A/G") Seq(((s.patientID, s.eventID, "rs591323_G"), 1.0), ((s.patientID, s.eventID, "rs591323_A"), 1.0))
      else if (s.testName == "rs591323" && s.value== "A/A") Seq(((s.patientID, s.eventID, "rs591323_G"), 0.0), ((s.patientID, s.eventID, "rs591323_A"), 2.0))

      // AC mutation
      else if (s.testName == "rs11724635" && s.value== "C/C") Seq(((s.patientID, s.eventID, "rs11724635_C"), 2.0), ((s.patientID, s.eventID, "rs11724635_A"), 0.0))
      else if (s.testName == "rs11724635" && s.value== "A/C") Seq(((s.patientID, s.eventID, "rs11724635_C"), 1.0), ((s.patientID, s.eventID, "rs11724635_A"), 1.0))
      else if (s.testName == "rs11724635" && s.value== "A/A") Seq(((s.patientID, s.eventID, "rs11724635_C"), 0.0), ((s.patientID, s.eventID, "rs11724635_A"), 2.0))

      // CT or TC mutation
      else if (s.testName == "rs71628662" && s.value== "C/T") Seq(((s.patientID, s.eventID, "rs71628662_C"), 1.0), ((s.patientID, s.eventID, "rs71628662_T"), 1.0))
      else if (s.testName == "rs71628662" && s.value== "T/T") Seq(((s.patientID, s.eventID, "rs71628662_C"), 0.0), ((s.patientID, s.eventID, "rs71628662_T"), 2.0))


      else if (s.testName == "rs118117788" && s.value== "C/C") Seq(((s.patientID, s.eventID, "rs118117788_C"), 2.0), ((s.patientID, s.eventID, "rs118117788_T"), 0.0))
      else if (s.testName == "rs118117788" && s.value== "C/T") Seq(((s.patientID, s.eventID, "rs118117788_C"), 1.0), ((s.patientID, s.eventID, "rs118117788_T"), 1.0))

      else if (s.testName == "rs8192591" && s.value== "C/C") Seq(((s.patientID, s.eventID, "rs8192591_C"), 2.0), ((s.patientID, s.eventID, "rs8192591_T"), 0.0))
      else if (s.testName == "rs8192591" && s.value== "C/T") Seq(((s.patientID, s.eventID, "rs8192591_C"), 1.0), ((s.patientID, s.eventID, "rs8192591_T"), 1.0))

      // GC mutation
      else if (s.testName == "rs114138760" && s.value== "G/G") Seq(((s.patientID, s.eventID, "rs114138760_G"), 2.0), ((s.patientID, s.eventID, "rs114138760_C"), 0.0))
      else if (s.testName == "rs114138760" && s.value== "C/G") Seq(((s.patientID, s.eventID, "rs114138760_G"), 1.0), ((s.patientID, s.eventID, "rs114138760_C"), 1.0))


      else Seq(((s.patientID, s.eventID, s.testName), s.value.toString.toDouble))
    }

    val ptauTuple = featureTuples.filter(s=>s._1._3 == "ptau").map(s=>((s._1._1, s._1._2), s._2)) // patientID, eventID, value
    val ttauTuple = featureTuples.filter(s=>s._1._3 == "ttau").map(s=>((s._1._1, s._1._2), s._2))
    val abetaTuple = featureTuples.filter(s=>s._1._3 == "abeta 1-42").map(s=>((s._1._1, s._1._2), s._2))

    val ptauTTauRatio = ptauTuple.join(ttauTuple).map(s=>((s._1._1, s._1._2, "ptauTTauRatio"), s._2._1/s._2._2)) // pTau/total Tau
    val ptauAbetaRatio = ptauTuple.join(abetaTuple).map(s=>((s._1._1, s._1._2, "ptauAbetaRatio"), s._2._1/s._2._2)) // pTau/abeta 1-42
    val ttauAbetaRatio = ttauTuple.join(abetaTuple).map(s=>((s._1._1, s._1._2, "ttauAbetaRatio"), s._2._1/s._2._2)) // tTau/abeta 1-42

    val featureTuplesPlus = featureTuples.union(ptauTTauRatio).union(ptauAbetaRatio).union(ttauAbetaRatio)


    val bioChemCount = featureTuplesPlus.map(s=>(s._1,1.0)).reduceByKey(_+_)
    val bioChemSum = featureTuplesPlus.reduceByKey(_+_)
    val bioChemfeatureTuples = bioChemCount.join(bioChemSum).map(s=>(s._1, s._2._2.toString.toDouble/s._2._1))

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

    val featureID = feature.map(s=>s._1._2).distinct().collect().sorted.zipWithIndex.toMap //(feature name -> feature ID)
    println("FeatureConstruction:construct start..")

    val numFeature = featureID.keys.size
    println("num of Features")
    println(numFeature)

    /** transform input feature */

    val result = feature.map(s=>(s._1._1, (s._1._2, s._2))).groupByKey().map{s=>
      val indexedFeatures = s._2.toList.map(s=>(featureID(s._1),s._2))
      val featureVector = Vectors.sparse(numFeature, indexedFeatures)
      val res = (s._1, featureVector)
      res
    }
    println("FeatureConstruction:construct ends..")
    result

  }



  /**
    * Given a feature tuples RDD, construct features in vector
    * format for each patient. feature name should be mapped
    * to some index and convert to dense feature format.
    * @param sc SparkContext to run
    * @param feature RDD of input feature tuples
    * @return
    */
  def constructDense(sc: SparkContext, feature: RDD[FeatureTuple], phenotypeLabel:RDD[(String, Int)]): RDD[(String, Vector)] = {

    /** save for later usage */
    feature.cache()
    //val scFeature = sc.broadcast(feature) //So we need this?


    val featureLog = feature.flatMap { s =>
      if (s._2 > 0) Seq(((s._1._1, s._1._2.concat("Log")), math.log(s._2)))
      else Seq()
    }


    /** create a feature name to id map*/

    val featureID = feature.map(s=>s._1._2).distinct().collect().sorted.zipWithIndex.toMap //(feature name -> feature ID)
    println("FeatureConstruction:constructDense start..")

    val numFeature = featureID.keys.size
    println("num of Features")
    println(numFeature)

    /** transform input feature */
    val result = feature.map(s=>(s._1._1, (s._1._2, s._2))).groupByKey().map{s=>
      val full_features = Array.fill(numFeature)(0.0)
      val indexedFeatures = s._2.toList.map(s=>(featureID(s._1),s._2)).toMap
      for (i <- 0 until numFeature) {
        full_features.update(i, indexedFeatures.getOrElse(i, 0.0))
      }
      val featureVector = Vectors.dense(full_features)
      val res = (s._1, featureVector)
      res
    }
    println("FeatureConstruction:constructDense ends..")
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

    // add log

    val featureLog = feature.flatMap { s =>
      if (s._2 > 0) Seq(((s._1._1, s._1._2, s._1._3.concat("Log")), math.log(s._2)))
      else Seq()
    }



    val featureAll = feature.union(featureLog)

    val featureMax = featureAll.map(s=>(s._1._3, s._2)).groupByKey().map(s=>(s._1, s._2.max))
    val featureMin = featureAll.map(s=>(s._1._3, s._2)).groupByKey().map(s=>(s._1, s._2.min))

    val featureRangeMin = featureMax.join(featureMin).map(s=>(s._1, (s._2._1-s._2._2, s._2._2)))
    val featureNorm = featureAll.map(s=>(s._1._3, (s._1._1, s._1._2, s._2))).join(featureRangeMin).
      map{s=>
        if (s._2._2._1 == 0) ((s._2._1._1, s._2._1._2, s._1), 0.0)
        else ((s._2._1._1, s._2._1._2, s._1), (s._2._1._3-s._2._2._2.toString.toDouble)/s._2._2._1.toString.toDouble)}


    // Generate training data for LSTM
    val featureID = featureNorm.map(s=>s._1._3).distinct().collect().sorted.zipWithIndex.toMap //(feature name -> feature ID)
    val timeID = Map(("SC", 0), ("BL", 0),
      ("V01", 3), ("V02", 6), ("V03", 9), ("V04", 12), ("V05", 18),
      ("V06", 24), ("V07", 30), ("V08", 36), ("V09", 42), ("V10", 48),
      ("V11", 54), ("V12", 60), ("V13", 72), ("V14", 84), ("V15", 96))

    import scala.collection.immutable.ListMap
    println("Feature Map")
    ListMap(featureID.toSeq.sortBy(_._2):_*).foreach(println)

    println("Generating training data for LSTM")

    import java.io._

    var pw = new PrintWriter(new File("output/withoutUPDRS_LSTM.train"))

    if (updrs == 0){
      pw = new PrintWriter(new File("output/withoutUPDRS_LSTM.train"))
    }
    else{
      pw = new PrintWriter(new File("output/withUPDRS_LSTM.train"))
    }


    val temp = featureNorm.map(s=>(s._1._1, (s._1._2, s._1._3, s._2))).groupByKey().map{s=>
      val indexedFeatures = s._2.toList.map(s=>(timeID(s._1), featureID(s._2),s._3))
      (s._1, indexedFeatures)
    }

    val temp2 = temp.join(phenotypeLabel).map(s=>(s._2._2.toString, s._2._1)).collect()

    for (patientFeature <- temp2){
      pw.printf(patientFeature._1)
      for (feat <- patientFeature._2.sortBy(s=> (s._1, s._2))){
        pw.print(" ")
        pw.print("(")
        pw.print(feat._1)
        pw.print(",")
        pw.print(feat._2)
        pw.print(")")
        pw.print(":")
        pw.print(feat._3)
      }
      pw.printf("\n")
    }

    pw.close()

    println("Done")

  }




  /** save features in pickle format for tflearn
    * @param sc SparkContext to run
    * @param feature RDD of input feature tuples
    * @return
    * */

  def saveDenseFeatures(sc: SparkContext, feature: RDD[(String, Vector)], phenotypeLabel:RDD[(String, Int)], updrs: Int): Unit = {

    /** save for later usage */
    println("saveDenseFeatures starts")

    feature.cache()

    val data = feature.sortBy(f => f._1).map(f => f._2)
    val labels = phenotypeLabel.sortBy(f => f._1).map(f => f._2)

    var data_file = new File("output/X_train1.csv")
    var labels_file = new File("output/Y_train1.csv")

    if(updrs == 0) {
      var data_file = new File("output/X_train0.csv")
      var labels_file = new File("output/Y_train0.csv")
    }
    else {
      var data_file = new File("output/X_train1.csv")
      var labels_file = new File("output/Y_train1.csv")
    }

    data_file.delete()
    labels_file.delete()

    data.map(f => f.toString.substring(1, f.toString.length-1)).saveAsTextFile(data_file.getAbsolutePath)
    labels.saveAsTextFile(labels_file.getAbsolutePath)



//    for(vector <- data) {
//      val vector_values = vector.toArray
//      val vector_length = vector_values.length
//      for(index <- 0  until  vector_length - 1) {
//        data_file.print(vector_values(index))
//        data_file.print(",")
//      }
//      data_file.print(vector_values(vector_length - 1))
//      data_file.printf("\n")
//    }
//
//    for(label <- labels) {
//      labels_file.printf(label.toString)
//      labels_file.printf("\n")
//    }
//
//    data_file.close()
//    labels_file.close()

    println("saveDenseFeatures Done")

  }
}


