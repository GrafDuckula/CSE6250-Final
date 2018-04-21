/**
 * @author Hang Su <hangsu@gatech.edu>.
 */

package edu.gatech.cse8803.model

import java.lang.annotation.Retention

case class EDU(patientID:String, years: Int)

case class DEMOGR(patientID:String, gender: String, age: Double)

case class BIOCHEM(patientID:String, eventID: String, testType: String, testName: String, value: Any, unit: String)

case class DATSCAN(patientID:String, eventID: String, CAUDATE: Double, PUTAMEN: Double, cpRatio: Double, caudateAsym: Double, putamenAsym: Double)

case class UPSIT(patientID:String, eventID: String, Score: Int)

case class REM(patientID:String, eventID: String, Score: Int)

case class MOCA(patientID:String, eventID: String, Score: Int)

case class UPDRS(patientID:String, eventID: String, Score: Int)

case class QUIP(patientID:String, eventID: String, Score: Int)

case class HVLT(patientID:String, eventID: String, immediateScore: Int, discrimScore: Int, retentScore: Double)

case class BLINE(patientID:String, eventID: String, Score: Double)

case class SFT(patientID:String, eventID: String, sumScore: Int, derivedSemScaled: Int, derivedSemT: Int)

case class LNSEQ(patientID:String, eventID: String, Score: Int, derivedScore: Int)

case class SDMOD(patientID:String, eventID: String, Score: Int, derivedTScore: Double)

case class ESS(patientID:String, eventID: String, Score: Int)

case class STAI(patientID:String, eventID: String, Score: Int, stateScore: Int, traitScore: Int)

case class GDS(patientID:String, eventID: String, Score: Int)

case class SCOPA(patientID:String, eventID: String, Score: Int)

case class HIST(patientID:String, eventID: String, Score: Double)