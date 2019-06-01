import Foundation

open class Debugging: NSObject {
    /* Debugging */
    
    /**
     Prints the time it took to complete the operation
     
     - Parameters:
     - title: Name of the operation (Default "unnamed")
     - operation: block of code that will be timed
     **/
    public class func printTimeElapsedWhenRunningCode(_ title: String = "unnamed", operation: ()->()) {
        let startTime = CFAbsoluteTimeGetCurrent()
        operation()
        let timeElapsed = CFAbsoluteTimeGetCurrent() - startTime
        print("Time elapsed for \(title): \(timeElapsed) s.")
    }
    
    /**
     Returns the time it took to complete the operation
     
     - Parameter operation: block of code that will be timed
     
     - Returns: A double of the time elapsed measured in seconds
     **/
    public class func timeElapsedInSecondsWhenRunningCode(operation: ()->()) -> Double {
        let startTime = CFAbsoluteTimeGetCurrent()
        operation()
        let timeElapsed = CFAbsoluteTimeGetCurrent() - startTime
        return Double(timeElapsed)
    }
}
