import Foundation
import Accelerate

/*
 Input
 - r (int), g (int), b (int)
 Output
 - int
 Train
 - Compute error: Check expected output, back propagate and adjust weights and bias
 */

public struct Color {
    let r: Int
    let g: Int
    let b: Int
    
    public func array () -> [Float] {
        return [Float(r), Float(g), Float(b)]
    }
}

struct Model {
    public static var fnc: ([Float], [Float]) -> (a: Float, z: [Float]) = { (input, weights) in
        let z: [Float] = VectorMath.multiply3(input, weights)
        /// a(z+b) b= bias
        return (Activation.relu(z.reduce(0, +)), z)
    }
    
    var layers: [Layer]
    
    public init () {
        layers = []
    }
}

struct Layer {
    private var size: Int
    
    var neurons: [Neuron]
    var error: Float
    
    public init (output size: Int) {
        self.neurons = []
        self.size = size
        self.error = 0
    }
    
    /// Returns the neurons activation output
    public func results () -> [Float] {
        return neurons.map({ $0.data.a })
    }
    
    public mutating func activate (_ input: [Float]) {
        for _ in 0..<self.size {
            var neuron = Neuron()
            neuron.data = Model.fnc(input, neuron.getWeights())
            print(neuron.data)
            
            neurons.append(neuron)
        }
    }
    
//    public mutating func xy () {
//
//    }
}

struct Neuron {
    private var weights: [Float]
    var data: (a: Float, z: [Float])
    
    public init () {
        self.weights = []
        self.data = (0, [])
    }
    
    public mutating func getWeights () -> [Float] {
        if weights.isEmpty { self.weights = self.randomizeWeights() }
        return self.weights
    }
    
    public func dA () -> Float {
        return VectorMath.derivative(of: Activation.relu, at: data.z.reduce(0, +))
    }
    
//    public func error (partial p: Float) -> [Float] {
//
//    }
    
    private func randomizeWeights (epsilon e: Float = 2) -> [Float] {
        let thetas = [
            Float.random(in: 10...11) * (2 * e) - e,
            Float.random(in: 5...11) * (2 * e) - e,
            Float.random(in: 1...11) * (2 * e) - e
        ]
        
        return thetas
    }
}

/** Calculate if the text color should be dark or light depending on the color of the background.
 Resources:
 - https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/neural_networks.html
 */
open class ColorNetwork: NSObject {
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
    
    private var model: Model!
    
    public override init () {
        super.init()
        
        self.model = Model()
    }
    
    /** Training
     1. Initialize weights randomly
     2. For each epoch
        - Epoch = a complete pass into all elements of your training set
     3. Do the forward propagation
        - You can calculate the output of the whole layer as a matrix multiplication followed by a element-wise activation function
     4. Calculate loss
     5. Do the backward propagation
     6. Update weights with Gradient descent (Optionally use gradient checking to verify backpropagation)
     7. Go to step 2 until you finish all epochs
     */
    public func train () {
        /// Step 2
        for color in self.trainingSet {
            /// Step 3 - Forward
            var hiddenLayer = Layer(output: 3)
            hiddenLayer.activate(color.array())
            
            /// Output
            var outputLayer = Layer(output: 1)
            outputLayer.activate(hiddenLayer.results())
            
            let expected = self.expectedOutput(color)
            print("h_theta(x)", outputLayer.results(), "expected", expected)
            
            /// Step 4 - Loss
            
            /// Step 5 - Backward
            self.backwards(result: outputLayer.results()[0], expected: expected, layer: hiddenLayer)
            /// Step 6 - Update weights
        }
    }
    
    private func backwards (result o: Float, expected e: Float, layer l: Layer) {
        let outError = e - o //p(3)
        
        var layerError: [Float] = []
        
        for neuron in l.neurons {
            var copy = neuron
            let vectorWeights = SIMD3(copy.getWeights())
            
            layerError = (vectorWeights * neuron.dA()).results()
        }
        
        print("output error:", outError, "layer 1 error:", layerError)
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
    
    private func expectedOutput (_ color: Color) -> Float {
        let weights: [Float] = [ 299, 587, 114 ]
        return VectorMath.dot(color.array(), weights)/1000
    }
    
    public func testing () {
        self.train()
    }
}
