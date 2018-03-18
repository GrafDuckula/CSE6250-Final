/**
 * @author Hang Su <hangsu@gatech.edu>.
 */

package edu.gatech.cse8803.model

case class EDU(patientID:String, years: Int)

case class DEMOGR(patientID:String, gender: String, age: Double)

case class BIOCHEM(patientID:String, eventID: String, testName: String, value: Any)

case class DATSCAN(patientID:String, eventID: String, CAUDATE: Double, PUTAMEN: Double)

case class UPSIT(patientID:String, eventID: String, Score: Double)

case class REM(patientID:String, eventID: String, Score: Int)

case class MOCA(patientID:String, eventID: String, Score: Int)

case class UPDRS(patientID:String, Score: Int)
