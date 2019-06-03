import Foundation
import Accelerate
import simd

public struct Color {
    let r: Float
    let g: Float
    let b: Float
    
    public func array () -> [Float] {
        return [(r/255), (g/255), (b/255)]
    }
    
    public func simd_float () -> SIMD3<Float> {
        return float3(self.array())
    }
}

/** Calculate if the text color should be dark or light depending on the color of the background.
 Resources:
 - https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/neural_networks.html
 */
open class ColorNetwork: NSObject {
    public typealias InputWeight = SIMD3<Float>
    /// h1, h2, h3
    public typealias IOutput = SIMD3<Float>
    /// h4, h5
    public typealias HOutput = SIMD2<Float>
    public typealias Output = (`in`: IOutput, h: HOutput, o: Float)
    
    private var trainingSet: [Color] = [
        Color(r: 0, g: 0, b: 0),
        Color(r: 255, g: 0, b: 0),
        Color(r: 0, g: 255, b: 0),
        Color(r: 0, g: 0, b: 255),
        Color(r: 255, g: 255, b: 0),
        Color(r: 255, g: 0, b: 255),
        Color(r: 0, g: 255, b: 255),
        Color(r: 255, g: 255, b: 255)
    ]
    
    var inputWeights = float3x3([
        float3(x: 0.1, y: 0.3, z: 0.24), // w1, w2, w3
        float3(x: 0.32, y: 0.5, z: 0.14), // w4, w5, w6
        float3(x: 0.27, y: 0.7, z: 0.63) // w7, w8, w9
    ])
    var inputBias = float3(x: 0.2, y: 0.5, z: 0.1)
    
    var hiddenWeights = float3x2([
        float2(x: 0.67, y: 0.28), // w10, w11
        float2(x: 0.21, y: 0.45), // w12, w13
        float2(x: 0.53, y: 0.55) // w14, w15
    ])
    var hiddenBias = float2(x: 0.25, y: 0.3)
    
    var outputWeights = float2(x: 0.62, y: 0.18) // w16, w17
    var outputBias: Float = 0.05
    
    /// Learning rate
    var n: Float = 0.5
    
    public func train () {
        var passes = 0
        var totalError: Float = 2
        let minError: Float = 0.000001
        
//        let color = trainingSet[7]
//        let expected = self.expectedOutput(color)
        
        while totalError > minError {
            for color in self.trainingSet {
                let expected = self.expectedOutput(color)
                
                // Forward Pass
                /// [0] = h1, [1] = h2, [2] = h3
                let output = self.forward(input: color)
                
                // Backwards Pass
                totalError = self.error(input: expected, output: output.o)
                
                let newWeights = self.backprop(input: color.simd_float(), results: output, expected: expected)
                
                // Update weights
                self.inputWeights = newWeights.in
                self.hiddenWeights = newWeights.hidden
                self.outputWeights = newWeights.out
                
                passes += 1
            }
        }
        
        print("Basic Network Training; Passes:", passes, "Error:", minError*100, "%")
        print("Final weights:\nInput:", self.inputWeights, "\nHiddne:", self.hiddenWeights, "\nOutput:", self.outputWeights)
//        print("actual:", forward(input: color).o, "expected:", expected)
    }
    
    public func apply (_ color: Color) -> Float {
        return self.forward(input: color).o
    }
    
    private func forward (input color: Color) -> Output {
        let a = self.inputForward(input: color)
        let a1 = self.hiddenForward(input: a)
        let output = self.outputForward(input: a1)
        
        return (a, a1, output)
    }
    
    private func inputForward (input color: Color) -> IOutput {
        let z = self.inputWeights * color.simd_float() + self.inputBias
        return Activation.relu_f3(z)
    }
    
    private func hiddenForward (input a: IOutput) -> HOutput {
        let z = a * hiddenWeights.transpose + self.hiddenBias
        return Activation.relu_f2(z)
    }
    
    private func outputForward (input a: HOutput) -> Float {
        let z = VectorMath.dot(a, self.outputWeights) + self.outputBias
        return Activation.relu(z)
    }
    
    private func backprop (input i: SIMD3<Float>, results r: Output, expected e: Float) -> (in: float3x3, hidden: float3x2, out: float2) {
        let dsdo = Activation.partialRelu(r.o)
        
        // Output Layer
        let loss = r.o-e
        let cost = loss*dsdo
        
        let pEwO = cost*r.h
        let wO = self.outputWeights-n*pEwO
        
        // Hidden 2 Layer
        let eo1os: float2 = loss * self.outputWeights
        let dsdh45: float2 = Activation.partialRelu_f2(r.h)
        
        let eoh45 = eo1os * dsdh45
        let pEh45 = float2x3([ (eoh45[0] * r.in), (eoh45[1] * r.in) ])
        let wHL2 = (self.hiddenWeights.transpose - n*pEh45).transpose // Transpose to be in the same order
        
        // Hidden 1 Layer
        let eht: float3 = eoh45 * self.hiddenWeights
        let dsdh = Activation.partialRelu_f3(r.in)
        let ehd = eht * dsdh
        let pEh = VectorMath.mul_f3x3(ehd, i)
        
        let wH: float3x3 = (self.inputWeights.transpose - n*pEh).transpose
        
        return (wH, wHL2, wO)
    }
    
    /** Genetic Algorithm
     1. Initial Population (Solution)
     2. Fitness function
     3. Selection
     4. Crossover
     5. Mutation
     */
    private func genetic () {
        
    }
    
    private func error (input e: Float, output o: Float) -> Float {
        return 0.5*pow(e - o, 2)
    }
    
    private func expectedOutput (_ color: Color) -> Float {
        let weights: [Float] = [ 299, 587, 114 ]
        return VectorMath.dot(color.array(), weights)/1000
    }
    
    public func testing () {
        for color in trainingSet {
            let expected = self.expectedOutput(color)
            
            print("actual", self.apply(color), "expected", expected)
        }
    }
}
