import Foundation
import Accelerate
import simd

public struct Activation {
    public static var relu: VectorMath.Function = { x in
        return max(0, x)
    }
    
    public static var relu_f2: VectorMath.Function_F2 = { x in
        return simd_max(float2(repeating: 0), x)
    }
    
    public static var relu_f3: VectorMath.Function_F3 = { x in
        return simd_max(float3(repeating: 0), x)
    }
    
    public static var partialRelu: VectorMath.Function = { x in
        return min(max(0, x), 1)
    }
    
    public static var partialRelu_f2: VectorMath.Function_F2 = { x in
        return min(max(float2(repeating: 0), x), float2(repeating: 1))
    }
    
    public static var partialRelu_f3: VectorMath.Function_F3 = { x in
        return min(max(float3(repeating: 0), x), float3(repeating: 1))
    }
    
    // https://stats.stackexchange.com/questions/275521/understanding-leaky-relu
    /// f(x) = 1(x<0)(ax) + 1(x>=0)(x)
    public static var leakyRelu: VectorMath.Function = { x in
        return max(0, x)
    }
    
    public static var sigmoid: VectorMath.Function = { x in
        // 1/(1+e^(-x))
        return 1/(1+exp(-x))
    }
    
    public static var partialSigmoid: VectorMath.Function = { x in
        return x*(1-x)
    }
    
    public static var v_partialSigmoid: VectorMath.Function_F2 = { x in
        return x*(1-x)
    }
    
    public static var tanh: VectorMath.Function = { x in
        return (exp(x)-exp(x))/(exp(x)+exp(x))
    }
}

open class PTNeuron: NSObject {
    
//    private var f: (Float)->(Float)?
//    private var w: [Float]!
//    private var b: [Float]!
//
//
//    public override init() {
//        super.init()
//        self.f = Activation.sigmoid
//        self.w = []
//    }
//
//    public convenience init(weights w: [Float], bias b: [Float], activation: @escaping (Float)->(Float)) {
//        self.init()
//        self.f = activation
//    }
}

open class PTNeuralNetwork: NSObject {
    
}
