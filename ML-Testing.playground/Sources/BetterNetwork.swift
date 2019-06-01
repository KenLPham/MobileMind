import Foundation
import Accelerate
import simd

/// Rewrite of BasicNetwork using vector math to do more concurrent computation
open class BetterNetwork: NSObject {
    var inputs: SIMD2<Float> = SIMD2([0.05, 0.1])
    
    var weights = float2x4(rows: [
        float2(x: 0.15, y: 0.2), // w1, w2
        float2(x: 0.25, y: 0.3), // w3, w4
        float2(x: 0.4, y: 0.45), // w5, w6
        float2(x: 0.5, y: 0.55) // w7, w8
    ])
    
    var bias: SIMD2<Float> = SIMD2([0.35, 0.6])
    var expected: SIMD2<Float> = [ 0.01, 0.99 ]
    
    /// Learning Multiplier
    var n: Float = 0.5
    
    public func train () {
        var totalError: Float = 2 // range [0-1]
        var passes = 0
        let minError: Float = 0.00005
        
        while totalError > minError {
            // [0] = hidden layer, [1] = output
            let output = self.apply(with: self.inputs)
            
            totalError = self.error(output: output[1])
            
            self.weights = self.backprop(results: output)
            
            passes += 1
        }
        
        print("Better Network Training; Passes:", passes, "Error:", minError*100, "%")
        print("Final weights:", self.weights)
    }
    
    private func forward (_ input: SIMD2<Float>, range r: ClosedRange<Int>, bias b: Int) -> SIMD2<Float> {
        var a: [Float] = []
        
        for i in r {
            let w = self.weights.transpose[i]
            a.append(self.activate(with: input, weights: w, bias: bias[b]))
        }
        
        return float2(a)
    }
    
    /// Backwards Propagation
    private func backprop (results o: matrix_float2x2) -> float2x4 {
        let output = o[1]
        
        let dsdo = Activation.v_partialSigmoid(o[1])
        let pEwN = (output-expected)*dsdo
        
        let newWeights56 = self.weights.transpose[2] - n * (pEwN[0]*o[0])
        let newWeights78 = self.weights.transpose[3] - n * (pEwN[1]*o[0])
        
        // get error of hidden layer
        let hiddenError = pEwN * self.weights[0].highHalf
        let hTotalError = VectorMath.sum_v2(hiddenError)
        
        let dsdh = Activation.v_partialSigmoid(o[0])
        let pEwH = hTotalError * dsdh
        
        // get new weights
        let newWeights12 = self.weights.transpose[0] - n * (pEwH[0]*inputs)
        let newWeights34 = self.weights.transpose[1] - n * (pEwH[1]*inputs)
        
        return float2x4(rows: [
            newWeights12,
            newWeights34,
            newWeights56,
            newWeights78
        ])
    }
    
    public func apply () -> float2 {
        return self.apply(with: self.inputs)[1]
    }
    
    public func apply (with input: [Float]) -> matrix_float2x2 {
        return self.apply(with: float2(input))
    }
    
    public func apply (with input: SIMD2<Float>) -> matrix_float2x2 {
        let hOut = self.forward(input, range: 0...1, bias: 0)
        let oOut = self.forward(hOut, range: 2...3, bias: 1)
        
        return float2x2([hOut, oOut])
    }
    
    private func activate (with i: SIMD2<Float>, weights w: SIMD2<Float>, bias b: Float) -> Float {
        let z = VectorMath.dot(w, i) + b
        return Activation.sigmoid(z)
    }
    
    /// Ew
    private func error (output o: SIMD2<Float>) -> Float {
        let et = expected-o
        return VectorMath.dot(float2(_simd_pow_d2(double2(et), double2(repeating: 2))), float2(repeating: 0.5))
    }
    
    public func testing () {
        print("output", self.apply(with: self.inputs), "expected", self.expected)
    }
}
