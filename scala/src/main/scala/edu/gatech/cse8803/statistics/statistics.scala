/**
  * @author Aiping Zheng <azheng39@gatech.edu>.
  */



package edu.gatech.cse8803.statistics

import edu.gatech.cse8803.model._
import org.apache.spark.rdd.RDD


//Print statistics for Proposal

package object statistics{

  def printStatistics(socEcoRDD: RDD[EDU], scDemogrEnrolledRDD: RDD[DEMOGR], bioChemRDD: RDD[BIOCHEM], datScanRDD: RDD[DATSCAN],quipRDD: RDD[QUIP],
                 upsitRDD: RDD[UPSIT], remRDD: RDD[REM], moCARDD: RDD[MOCA], updrsIRDD: RDD[UPDRS], updrsIIRDD: RDD[UPDRS],
                 patientsWithLabel: RDD[(String, Int)]): Unit= {


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

  }
}
