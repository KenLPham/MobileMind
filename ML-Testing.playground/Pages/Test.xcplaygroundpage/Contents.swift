//: [Previous](@previous)

import Foundation

var times: [Double] = []

// Basic Average Time: 0.023460209369659424

//for _ in 0..<10 {
//    let basic = BasicNetwork()
//    let t = Debugging.timeElapsedInSecondsWhenRunningCode {
//        basic.train()
//    }
//    times.append(t)
//}

// Better Average Time: 0.019561493396759035

//for _ in 0..<10 {
//    let better = BetterNetwork()
//    let t = Debugging.timeElapsedInSecondsWhenRunningCode {
//        better.train()
//    }
//    times.append(t)
//}

// Advanced Average Time: 0.09434828758239747

//for _ in 0..<10 {
//    let better = AdvancedNetwork()
//    let t = Debugging.timeElapsedInSecondsWhenRunningCode {
//        better.train()
//    }
//    times.append(t)
//}


let average = times.reduce(0, +)/10
print(average)

//let network = BetterNetwork()
//network.train()
// Advanced: [[-3.6102164, -4.3346467, 2.5957444, 3.472063], [-3.627436, -4.3239384, 2.5738342, 3.483712]] Passes: 18709
// Basic: [[0.3712084, 0.64241594, 0.46997926, 0.7399576], [-3.837633, -3.812755, 2.8062522, 2.8698068]] Passed: 8257
// Better: simd_float4x2([[0.3712084, 0.64241594], [0.46997926, 0.7399576], [-3.837633, -3.812755], [2.8062522, 2.8698068]])
