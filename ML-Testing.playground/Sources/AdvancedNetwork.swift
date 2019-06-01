import Foundation
import Accelerate
import simd

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
//        var totalError: Float = 2
//        var passes = 0
//        var minError: Float = 0.00005
        
//        while totalError > minError {
//            /** Forward Pass */
//        }
        
        let output = self.forward()
        print(output)
    }
    
    private func forward () -> ([Float], [Float]) {
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
}
