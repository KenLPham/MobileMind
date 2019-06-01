//: [Previous](@previous)

import Foundation

var times: [Double] = []

// Basic Average Time: 0.023460209369659424

for _ in 0..<10 {
    let basic = BasicNetwork()
    let t = Debugging.timeElapsedInSecondsWhenRunningCode {
        basic.train()
    }
    times.append(t)
}

// Better Average Time: 0.019561493396759035

//for _ in 0..<10 {
//    let better = BetterNetwork()
//    let t = Debugging.timeElapsedInSecondsWhenRunningCode {
//        better.train()
//    }
//    times.append(t)
//}

let average = times.reduce(0, +)/10
print(average)
