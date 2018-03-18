package edu.gatech.cse8803.clustering

/**
  * @author Hang Su <hangsu@gatech.edu>
  */


import breeze.linalg.{sum, DenseMatrix => BDM, DenseVector => BDV, CSCMatrix => BSM, Matrix => BM}
import breeze.linalg._
import breeze.numerics._
import org.apache.spark.mllib.linalg.{DenseMatrix, Matrix, SparseMatrix, Vector, Vectors}
import org.apache.spark.mllib.linalg.distributed.RowMatrix



object NMF {

  /**
   * Run NMF clustering 
   * @param V The original non-negative matrix 
   * @param k The number of clusters to be formed, also the number of cols in W and number of rows in H
   * @param maxIterations The maximum number of iterations to perform
   * @param convergenceTol The maximum change in error at which convergence occurs.
   * @return two matrixes W and H in RowMatrix and DenseMatrix format respectively 
   */
  def run(V: RowMatrix, k: Int, maxIterations: Int, convergenceTol: Double = 1e-4): (RowMatrix, BDM[Double]) = {

    /**
     * TODO 1: Implement your code here
     * Initialize W, H randomly 
     * Calculate the initial error (Euclidean distance between V and W * H)
     */

    var W = new RowMatrix(V.rows.map(_ => BDV.rand[Double](k)).map(fromBreeze).cache)
    val H = BDM.rand[Double](k, V.numCols().toInt)
    val dist = EucDistance(V, W, H)

    /**
     * TODO 2: Implement your code here
     * Iteratively update W, H in a parallel fashion until error falls below the tolerance value 
     * The updating equations are, 
     * H = H.* W^T^V ./ (W^T^W H)
     * W = W.* VH^T^ ./ (W H H^T^)
     */
    var oldDist = 1.0
    var newDist = dist
    var errChange = 1.0
    var Hs = H*H.t
    var Ws = computeWTV(W, W)

    for (n <- 1 to maxIterations if errChange > convergenceTol) {
      V.rows.cache()
      W.rows.cache()

      //update W
      Hs = H*H.t
      W = dotProd(W, dotDiv(multiply(V, H.t), multiply(W, Hs)))

      //update H
      Ws = computeWTV(W, W)
      H:*=(computeWTV(W, V):/(Ws*H).mapValues(_ + 2.0e-15))

      oldDist = newDist
      newDist = EucDistance(V, W, H)
      errChange = abs(oldDist-newDist)/newDist

      println(f"Iteration: [$n%d]")
      println(f"Error change is: $errChange%.5f")
      println(f"oldDist: $oldDist%.5f")
      println(f"newDist: $newDist%.5f")

    }

    W.rows.unpersist()
    V.rows.unpersist()

    (W, H)
  }


  /**  
  * RECOMMENDED: Implement the helper functions if you needed
  * Below are recommended helper functions for matrix manipulation
  * For the implementation of the first three helper functions (with a null return), 
  * you can refer to dotProd and dotDiv whose implementation are provided
  */
  /**
  * Note:You can find some helper functions to convert vectors and matrices
  * from breeze library to mllib library and vice versa in package.scala
  */

  /** compute the mutiplication of a RowMatrix and a dense matrix */
  def multiply(X: RowMatrix, d: BDM[Double]): RowMatrix = {
    X.multiply(fromBreeze_mod(d))
  }

 /** get the dense matrix representation for a RowMatrix */
  def getDenseMatrix(X: RowMatrix): BDM[Double] = {
    BDM(X.rows.collect().map(_.toArray):_*)
  }
  // what does :_* mean? Unpacking

  /** matrix multiplication of W.t and V */
  def computeWTV(W: RowMatrix, V: RowMatrix): BDM[Double] = {
    getDenseMatrix(W).t*getDenseMatrix(V)
  }

  /** dot product of two RowMatrixes */
  def dotProd(X: RowMatrix, Y: RowMatrix): RowMatrix = {
    val rows = X.rows.zip(Y.rows).map{case (v1: Vector, v2: Vector) =>
      toBreezeVector(v1) :* toBreezeVector(v2)
    }.map(fromBreeze)
    new RowMatrix(rows)
  }

  /** dot division of two RowMatrixes */
  def dotDiv(X: RowMatrix, Y: RowMatrix): RowMatrix = {
    val rows = X.rows.zip(Y.rows).map{case (v1: Vector, v2: Vector) =>
      toBreezeVector(v1) :/ toBreezeVector(v2).mapValues(_ + 2.0e-15)
    }.map(fromBreeze)
    new RowMatrix(rows)
  }

  def dotSub(X: RowMatrix, Y: RowMatrix): RowMatrix = {
    val rows = X.rows.zip(Y.rows).map{case (v1: Vector, v2: Vector) =>
      toBreezeVector(v1) :- toBreezeVector(v2)
    }.map(fromBreeze)
    new RowMatrix(rows)
  }


/*  def EucDistance(V: RowMatrix, W: RowMatrix, H: BDM[Double]): Double = {
    val WH = multiply(W, H)
    val dist = getDenseMatrix(dotSub(V, WH))
    0.5*sqrt(sum(dist:*dist))
  }*/

/*  def EucDistance(V: RowMatrix, W: RowMatrix, H: BDM[Double]): Double = {
    val WH = multiply(W, H)
    val dist = dotSub(V, WH)
    val rows = dist.rows.map{case (v1: Vector) =>toBreezeVector(v1).map(s=>s*s)}.map(fromBreeze)
    val A = new RowMatrix(rows).rows.collect().map(s=>s.toArray.sum)
    0.5*sqrt(A.sum)
  }*/

  def EucDistance(V: RowMatrix, W: RowMatrix, H: BDM[Double]): Double = {
    val WH = multiply(W, H)
    val dist = getDenseMatrix(dotSub(V, WH))
    0.5*sqrt(sum(dist.mapValues(d => d * d)))
  }

  def fromBreeze_mod(breeze: BM[Double]): Matrix = {
    breeze match {
      case dm: BDM[Double] =>
        new DenseMatrix(dm.rows, dm.cols, dm.toArray)
      case sm: BSM[Double] =>
        // There is no isTranspose flag for sparse matrices in Breeze
        new SparseMatrix(sm.rows, sm.cols, sm.colPtrs, sm.rowIndices, sm.data)
      case _ =>
        throw new UnsupportedOperationException(
          s"Do not support conversion from type ${breeze.getClass.getName}.")
    }
  }
}

