import Foundation
import Accelerate
import simd

open class VectorMath: NSObject {
    public typealias Function = (Float)->(Float)
    public typealias Function_D = (Double)->(Double)
    public typealias Function_F2 = (SIMD2<Float>)->(SIMD2<Float>)
    public typealias Function_F3 = (SIMD3<Float>)->(SIMD3<Float>)
    
    public class func norm_d (_ a: [Double]) -> Double {
        var n = Int32(a.count)
        var results = [Double](repeating: 0, count: a.count)
        let power = [Double](repeating: 2, count: a.count)
        
        vvpow(&results, power, a, &n)
        
        return sqrt(results.reduce(0, +))
    }
    
    /// - returns: dot product of a * b (Double Percision)
    public class func dot_d (_ a: [Double], _ b: [Double]) -> Double {
        var sum: Double = 0.0
        vDSP_dotprD(a, 1, b, 1, &sum, vDSP_Length(a.count))
        return sum
    }
    
    public class func dot (_ a: [Float], _ b: [Float]) -> Float {
        var sum: Float = 0.0
        vDSP_dotpr(a, 1, b, 1, &sum, vDSP_Length(a.count))
        return sum
    }
    
    public class func dot (_ a: SIMD2<Float>, _ b: SIMD2<Float>) -> Float {
        return simd_dot(a, b)
    }
    
    public class func dot (_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Float {
        return simd_dot(a, b)
    }
    
    public class func similarity_d (_ a: [Double], _ b: [Double]) -> Double {
        return dot_d(a, b) / (norm_d(a) * norm_d(b))
    }
    
    // Calculate derivative of double percision functions
    public class func derivative_d (of f: Function_D, at x: Double) -> Double {
        let h: Double = 1e-12
        return (f(x+h) - f(x))/h
    }
    
    public class func derivative (of f: Function, at x: Float) -> Float {
        let h: Float = 1e-6
        return (f(x+h) - f(x))/h
    }
    
    /// Element Wise Multiplication for an array of size 3
    public class func multiply3 (_ a: [Float], _ b: [Float]) -> [Float] {
        let vectorA = SIMD3(a)
        let vectorB = SIMD3(b)
        return (vectorA * vectorB).results()
    }
    
    public class func mul_f3x2 (_ a: SIMD2<Float>, _ b: float3x2) -> float3x2 {
        let x = a[0] * b.transpose[0]
        let y = a[1] * b.transpose[1]
        
        return float3x2(rows: [x, y])
    }
    
    public class func mul_f3x3 (_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> float3x3 {
        let x = a[0] * b
        let y = a[1] * b
        let z = a[2] * b
        
        return float3x3(columns: (x, y, z))
    }
    
    public class func sum_v2 (_ a: SIMD2<Float>) -> Float {
        return self.dot(a, float2(repeating: 1))
    }
    
    public class func sum_f3 (_ a: SIMD3<Float>) -> Float {
        return self.dot(a, float3(repeating: 1))
    }
}

extension SIMD3 {
    func results () -> [Scalar] {
        return [self.x, self.y, self.z]
    }
}
