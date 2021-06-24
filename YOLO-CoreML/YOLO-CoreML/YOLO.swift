import Foundation
import UIKit
import CoreML

class YOLO {
  /**
   Image size can be extracted from a Yolo v5 models with the following:
   grid_size = detect.grid[0].shape[2:4]
   stride = detect.stride[0].numpy()
   print(f'Input size: {grid_size * stride}')
   */
  public static let inputWidth = 640
  public static let inputHeight = 640

  // Tweak these values to get more or fewer predictions.
  public static let maxBoundingBoxes = 10
  let confidenceThreshold: Float = 0.25
  let iouThreshold: Float = 0.45
  
  /**
   Anchor + Stride can be extract from a Yolo v5 model with the following:
   from models.yolo import Detect
   for module in model.modules():
       if isinstance(module, Detect):
           print(f'Anchors: {module.anchor_grid.squeeze().tolist()}')
           print(f'Stride: {module.stride.tolist()}')
   */
  let anchors: [[[Float]]] = [
      [[12.07812, 10.70312], [17.35938, 16.76562], [16.32812, 31.57812]],
      [[27.20312, 20.84375], [30.59375, 32.90625], [50.37500, 26.43750]],
      [[47.00000, 43.78125], [37.96875, 69.75000], [75.37500, 67.93750]],
  ]
  let strides: [Float] = [8, 16, 32]
  let model = yolo()
  
  struct Prediction {
    let classIndex: Int
    let score: Float
    let rect: CGRect
  }

  public init() { }

  public func predict(image: CVPixelBuffer) throws -> [Prediction] {
    if let output = try? model.prediction(image: image) {
      return computeBoundingBoxes(layers: [output._714, output._727, output._740])
    } else {
      return []
    }
  }

  public func computeBoundingBoxes(layers: [MLMultiArray]) -> [Prediction] {
    var predictions = [Prediction]()
    for (layer, features) in layers.enumerated() {
      convertFeaturestoPredictions(features: features, predictions: &predictions, layer: layer)
    }
    return nonMaxSuppression(boxes: predictions, limit: YOLO.maxBoundingBoxes, threshold: iouThreshold)
  }
  
  private func convertFeaturestoPredictions(features: MLMultiArray, predictions: inout [Prediction], layer: Int) {
    assert(features.shape[0].intValue == 1)  // Batch size of 1
    let boxesPerCell = features.shape[1].intValue
    let gridHeight = features.shape[2].intValue
    let gridWidth = features.shape[3].intValue
    let numClasses = features.shape[4].intValue - 5

    let boxStride = features.strides[1].intValue
    let yStride = features.strides[2].intValue
    let xStride = features.strides[3].intValue
    assert(features.strides[4].intValue == 1)  // The below code assumes a channel stride of 1.
    let gridSize = strides[layer]

    assert(features.dataType == MLMultiArrayDataType.float32) // Ensure 32 bit before using unsafe pointer.
    let featurePointer = UnsafeMutablePointer<Float32>(OpaquePointer(features.dataPointer))

    for b in 0..<boxesPerCell {
      let anchorW = anchors[layer][b][0]
      let anchorH = anchors[layer][b][1]
      for cy in 0..<gridHeight {
        for cx in 0..<gridWidth {
          let d = b*boxStride + cx*xStride + cy*yStride
          let tc = Float(featurePointer[d + 4])
          let confidence = sigmoid(tc)

          var classes = [Float](repeating: 0, count: numClasses)
          for c in 0..<numClasses {
            classes[c] = Float(featurePointer[d + 5 + c])
          }
          classes = softmax(classes)

          let (detectedClass, bestClassScore) = classes.argmax()
          let confidenceInClass = bestClassScore * confidence
          if confidenceInClass > confidenceThreshold {
            let tx = Float(featurePointer[d])
            let ty = Float(featurePointer[d + 1])
            let tw = Float(featurePointer[d + 2])
            let th = Float(featurePointer[d + 3])

            // Code converted from:
            // https://github.com/ultralytics/yolov5/blob/ae4261c7749ff644f45c66b79ecb1fff06437052/models/yolo.py
            // Inside Detect.forward
            let x = (sigmoid(tx) * 2 - 0.5 + Float(cx)) * gridSize
            let y = (sigmoid(ty) * 2 - 0.5 + Float(cy)) * gridSize
            let w = pow(sigmoid(tw) * 2.0, 2) * anchorW
            let h = pow(sigmoid(th) * 2.0, 2) * anchorH

            let rect = CGRect(
                x: CGFloat(x - w/2),
                y: CGFloat(y - h/2),
                width: CGFloat(w),
                height: CGFloat(h)
            )

            let prediction = Prediction(
                classIndex: detectedClass,
                score: confidenceInClass,
                rect: rect
            )
            predictions.append(prediction)
          }
        }
      }
    }
  }
}
