Iterative Hard Thresolding
==========================

This is a translation to Python of the iterative hard thresholding algorithm of
Blumensath & Davies.

Description

   Implements the M-sparse algorithm described in [1], [2] and [3].
   This algorithm takes a gradient step and then thresholds to only retain
   M non-zero elements. It allows the step-size to be calculated
   automatically as described in [3] and is therefore now independent from 
   a rescaling of P.
   
   
 References
   [1]  T. Blumensath and M.E. Davies, "Iterative Thresholding for Sparse 
        Approximations", submitted, 2007
   [2]  T. Blumensath and M. Davies; "Iterative Hard Thresholding for 
        Compressed Sensing" to appear Applied and Computational Harmonic 
        Analysis 
   [3]  T. Blumensath and M. Davies; "A modified Iterative Hard 
        Thresholding algorithm with guaranteed performance and stability" 
        in preparation (title may change) 
