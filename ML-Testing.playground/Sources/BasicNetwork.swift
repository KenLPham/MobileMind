import Foundation

/// Resources:
/// - https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
open class BasicNetwork: NSObject {
    public typealias Output = (Float, Float)
    
    var inputs: [Float] = [0.05, 0.1]
    var weights: [[Float]] = [
        /// Layer 1 w1->w4
        [ 0.15, 0.2, 0.25, 0.3 ],
        /// Layer 2 w5->w8
        [ 0.4, 0.45, 0.5, 0.55 ]
    ]
    
    var bias: [Float] = [ 0.35, 0.6 ]
    
    var expected: [Float] = [ 0.01, 0.99 ]
    
    /// Learning Multiplier
    var n: Float = 0.5
    
    public func train () {
        var totalError: Float = 2
        var passes = 0
        let minError: Float = 0.00005
        
        while totalError > minError {
            /** Forward Pass */
            
            /// Layer 1
            let hiddenLayerOut = self.calculateLayerOne()
            
            /// Output
            let outputLayerOut = self.calculateOutputLayer(output: hiddenLayerOut)
            
            /** Backwards pass */
            
            /// Total Error
            totalError = self.error(output: outputLayerOut)
            
            /// Get new weights for Output 1
            let dsdo1 = Activation.partialSigmoid(outputLayerOut.0) // 0.1868156
            
            let pEw5: Float = (outputLayerOut.0 - expected[0]) * dsdo1 * hiddenLayerOut.a1 // 0.082167044
            let newW5: Float = self.weights[1][0] - n*pEw5 // 0.35891648
            
            let pEw6: Float = (outputLayerOut.0 - expected[0]) * dsdo1 * hiddenLayerOut.a2
            let newW6: Float = self.weights[1][1] - n*pEw6 // 0.408666186
            
            /// Get new weights for Output 2
            let dsdo2 = Activation.partialSigmoid(outputLayerOut.1) // 0.17551005
            
            let pEw7: Float = (outputLayerOut.1 - expected[1]) * dsdo2 * hiddenLayerOut.a1
            let newW7: Float = self.weights[1][2] - n*pEw7 // 0.5113013
            
            let pEw8: Float = (outputLayerOut.1 - expected[1]) * dsdo2 * hiddenLayerOut.a2
            let newW8: Float = self.weights[1][3] - n*pEw8 // 0.56137013
            
            let newWeights1: [Float] = [ newW5, newW6, newW7, newW8 ]
            
            /// Error of neuron 1 with output 1
            let pEo1: Float = outputLayerOut.0 - expected[0] // 0.7413651
            let eo1no1: Float = pEo1 * dsdo1 // 0.13849856
            
            let eo1o1: Float = eo1no1 * weights[1][0] // w5; 0.055399425
            
            /// Error of neuron 1 with output 2
            let pEo2: Float = outputLayerOut.1 - expected[1]
            let eo2no2: Float = pEo2 * dsdo2
            
            let eo2o2: Float = eo2no2 * weights[1][2] // w7; -0.019049117
            
            /// Error of neuron 1
            let etn1: Float = eo1o1 + eo2o2 // 0.03635031
            
            /// Get weights for Hidden Neuron 1
            let dsdh1 = Activation.partialSigmoid(hiddenLayerOut.a1) // 0.2413007
            
            let pEw1: Float = etn1 * dsdh1 * inputs[0] // 0.00043856777
            let newW1: Float = self.weights[0][0] - n*pEw1 // 0.14978072
            
            let pEw2: Float = etn1 * dsdh1 * inputs[1]
            let newW2: Float = self.weights[0][1] - n*pEw2 // 0.19956143
            
            /// Error of neuron 2 with output 1
            let pEo3: Float = outputLayerOut.0 - expected[0]
            let eo3no3: Float = pEo3 * dsdo1
            
            let eo3o3: Float = eo3no3 * weights[1][1]
            
            /// Error of neuron 2 with output 2
            let pEo4: Float = outputLayerOut.1 - expected[1]
            let eo4no4: Float = pEo4 * dsdo2
            
            let eo4o4: Float = eo4no4 * weights[1][3]
            
            /// Error of neuron 2
            let etn2: Float = eo3o3 + eo4o4
            
            /// Get weights for Hidden Neuron 2
            let dsdh2 = Activation.partialSigmoid(hiddenLayerOut.a2)
            
            let pEw3: Float = etn2 * dsdh2 * inputs[0]
            let newW3: Float = self.weights[0][2] - n*pEw3 // 0.24975114
            
            let pEw4: Float = etn2 * dsdh2 * inputs[1]
            let newW4: Float = self.weights[0][3] - n*pEw4 // 0.29950229
            
            let newWeights0: [Float] = [ newW1, newW2, newW3, newW4 ]
            
            self.weights[0] = newWeights0
            self.weights[1] = newWeights1
            
            passes += 1
        }
        
        print("Basic Network Training; Passes:", passes, "Error:", minError*100, "%")
        print("Final weights:", self.weights)
    }
    
    public func apply () -> Output {
        /// Layer 1
        let hiddenLayerOut = self.calculateLayerOne()

        /// Output
        let outputLayerOut = self.calculateOutputLayer(output: hiddenLayerOut)
        
        return outputLayerOut
    }
    
    /// Calculate z (net H) and a (out H) of the hidden layer
    private func calculateLayerOne () -> (a1: Float, a2: Float) {
        let wH1 = self.weights[0] // w1->w4
        
        /// H1
        let z1 = wH1[0]*inputs[0] + wH1[1]*inputs[1] + bias[0]*1 // net H1 = 0.3775
        let a1 = Activation.sigmoid(z1) // outH1 = 0.593269992
        
        /// H2
        let z2 = wH1[2]*inputs[0] + wH1[3]*inputs[1] + bias[0]*1 // net H2 = 0.39249998
        let a2 = Activation.sigmoid(z2) // outH2 = 0.596884378
        
        return (a1, a2)
    }
    
    private func calculateOutputLayer (output o: (a1: Float, a2: Float)) -> Output {
        let w = self.weights[1] // w5 -> w8
        
        /// O1
        let z1 = w[0]*o.a1 + w[1]*o.a2 + bias[1]*1 // net O1 = 1.105905967
        let a1 = Activation.sigmoid(z1) // out O1 = 0.75136507
        
        /// O2
        let z2 = w[2]*o.a1 + w[3]*o.a2 + bias[1]*1 // net O2 = 0.772928465
        let a2 = Activation.sigmoid(z2) // out O2 = 0.772928465
        
        return (a1, a2)
    }
    
    /// Calculate Total Error
    private func error (output o: Output) -> Float {
        let e1 = 0.5 * pow(expected[0]-o.0, 2) // 0.274811083
        let e2 = 0.5 * pow(expected[1]-o.1, 2) // 0.023560026
        let et = e1+e2 // 0.298371109
        
        return et
    }
    
    public func testing () {
        print("output", self.apply(), "expected", self.expected)
    }
}
