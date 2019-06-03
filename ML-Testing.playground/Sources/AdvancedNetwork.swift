import Foundation
import Accelerate
import simd

/// BNNS set up with back propagation. It seems to take way more passes than the other networks and ends up with drastically different weights, but it does end up getting the right answer
open class AdvancedNetwork: NSObject {
    private var filter: BNNSFilter?
    private var outFilter: BNNSFilter?
    
    var inputs: [Float] = [0.05, 0.1]
    
    private var weights: [[Float]] = [
        /// Layer 1 w1->w4
        [ 0.15, 0.2, 0.25, 0.3 ],
        /// Layer 2 w5->w8
        [ 0.4, 0.45, 0.5, 0.55 ]
    ]
    
    private var bias: [Float] = [ 0.35, 0.6 ]
    
    var expected: [Float] = [ 0.01, 0.99 ]
    
    /// Learning Multiplier
    private var n: Float = 0.5
    
    public override init() {
        super.init()
        self.updateFilter()
    }
    
    public func train () {
        var totalError: Float = 2
        var passes = 0
        let minError: Float = 0.00005
        
        while totalError > minError {
            /** Forward Pass */
            let output = self.forward()
            
            self.cleanup()
            
            /** Back propagation */
            totalError = self.error(output: output.1)
            
            let dsdo: [Float] = [Activation.partialSigmoid(output.1[0]),
                                 Activation.partialSigmoid(output.1[1])]
            let t: [Float] = [ ((output.1[0]-expected[0]) * dsdo[0]), // 5,6
                ((output.1[1]-expected[1]) * dsdo[1]) ] // 7,8
            let pEwO: [Float] = [
                (t[0] * output.0[0]), (t[0] * output.0[1]), // 5, 6
                (t[1] * output.0[0]), (t[1] * output.0[1]) // 7, 8
            ]
            
            let oWeights: [Float] = [
                (self.weights[1][0]-n*pEwO[0]), (self.weights[1][1]-n*pEwO[1]), // w5, w6
                (self.weights[1][2]-n*pEwO[2]), (self.weights[1][3]-n*pEwO[3]) // w7, w8
            ]
            
            let pEwH: [Float] = [ ((output.1[0]-expected[0]) * dsdo[0]),
                                  ((output.1[1]-expected[1]) * dsdo[1]) ]
            
            let h1Error = (pEwH[0] * weights[1][0]) + (pEwH[1] * weights[1][2])
            let h2Error = (pEwH[0] * weights[1][1]) + (pEwH[1] * weights[1][3])
            
            let dsdh: [Float] = [Activation.partialSigmoid(output.0[0]),
                                 Activation.partialSigmoid(output.0[1])]
            
            let pEwI: [Float] = [
                (h1Error * dsdh[0] * inputs[0]), (h1Error*dsdh[0] * inputs[1]),
                (h2Error * dsdh[1] * inputs[0]), (h2Error*dsdh[1] * inputs[1])
            ]
            
            let hWeights: [Float] = [
                (self.weights[0][0]-n*pEwI[0]), (self.weights[0][1]-n*pEwI[1]),
                (self.weights[0][2]-n*pEwI[2]), (self.weights[0][3]-n*pEwI[3])
            ]
            
            self.weights = [oWeights, hWeights]
            
            passes += 1
        }
        
        print("Advanced Network Training; Passes:", passes, "Error:", minError*100, "%")
        print("Final weights:", self.weights)
        
        print("Results:", self.forward(), "Expected:", self.expected)
    }
    
    private func forward () -> ([Float], [Float]) {
        if self.filter == nil || self.outFilter == nil {
            self.updateFilter()
        }
        
        var hOutput: [Float] = [ 0, 0 ]
        var output: [Float] = [ 0, 0 ]
        
        if BNNSFilterApply(self.filter, inputs, &hOutput) != 0 {
            print("Failed to apply hidden filter")
        }
        
        if BNNSFilterApply(self.outFilter, hOutput, &output) != 0 {
            print("Failed to apply output filter")
        }
        
        return (hOutput, output)
    }
    
    public func updateFilter () {
        let sigmoid = BNNSActivation(function: BNNSActivationFunction.sigmoid)
        
        let hW: [Float] = self.weights[0]
        let hB = [Float](repeating: bias[0], count: 2)
        
        let hWeights = BNNSLayerData(data: hW, data_type: BNNSDataType.float)
        let hBias = BNNSLayerData(data: hB, data_type: BNNSDataType.float)
        
        var hParam = BNNSFullyConnectedLayerParameters(in_size: 2, out_size: 2, weights: hWeights, bias: hBias, activation: sigmoid)
        
        var iDescr = BNNSVectorDescriptor(size: 2, data_type: BNNSDataType.float)
        var hDescr = BNNSVectorDescriptor(size: 2, data_type: BNNSDataType.float)
        
        self.filter = BNNSFilterCreateFullyConnectedLayer(&iDescr, &hDescr, &hParam, nil)
        
        let oW: [Float] = self.weights[1]
        let oB = [Float](repeating: bias[1], count: 2)
        
        let oWeights = BNNSLayerData(data: oW, data_type: BNNSDataType.float)
        let oBias = BNNSLayerData(data: oB, data_type: BNNSDataType.float)
        
        var oParam = BNNSFullyConnectedLayerParameters(in_size: 2, out_size: 2, weights: oWeights, bias: oBias, activation: sigmoid)
        
        /// - todo: test to see if one descr can be used without causing an issue
        var aDescr = BNNSVectorDescriptor(size: 2, data_type: BNNSDataType.float)
        var oDescr = BNNSVectorDescriptor(size: 2, data_type: BNNSDataType.float)
        
        self.outFilter = BNNSFilterCreateFullyConnectedLayer(&aDescr, &oDescr, &oParam, nil)
    }
    
    public func cleanup () {
        BNNSFilterDestroy(self.filter)
        BNNSFilterDestroy(self.outFilter)
        
        self.filter = nil
        self.outFilter = nil
    }
    
    private func error (output o: [Float]) -> Float {
        let e1 = 0.5 * pow(expected[0]-o[0], 2)
        let e2 = 0.5 * pow(expected[1]-o[1], 2)
        
        return e1+e2
    }
    
    deinit {
        self.cleanup()
    }
}
