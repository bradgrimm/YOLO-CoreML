# YOLO with Core ML

__If you want to use this code. You will need to update a few variables for your model. See YOLO.swift and Helpers.swift for more details.__

This repo was forked and modified from [syshen/YOLO-CoreML](https://github.com/syshen/YOLO-CoreML). Some changes I made: 

1. Updated to work with [Yolo v5](https://github.com/ultralytics/yolov5).
2. Added comments that explain how to grab stride, anchors, and other data from an exported model. 
3. Updated the code to work on all output layers instead of just the first one.

<b>This works on their model as of June 23, 2021.</b>

A working example model can be found [here](https://github.com/bradgrimm/YOLO-CoreML/blob/master/Models/2021_06_21.pt). But do note that this model was trained to predict Lego pieces not the same labels as in syshen. And there have been changes in this repository that aren't compatible with syshen's models:
1. Label names have been changed in [Helpers.swift](https://github.com/bradgrimm/YOLO-CoreML/blob/master/YOLO-CoreML/YOLO-CoreML/Helpers/Helpers.swift )
2. Layers are named differently (now they are _714, _727, _740) in [Yolo.swift](https://github.com/bradgrimm/YOLO-CoreML/blob/master/YOLO-CoreML/YOLO-CoreML/YOLO.swift)

At a minimum you will likely have to change the label names to be compatible with your model.

## About YOLO object detection

YOLO is an object detection network. It can detect multiple objects in an image and puts bounding boxes around these objects. [Read hollance's blog post about YOLO](http://machinethink.net/blog/object-detection-with-yolo/) to learn more about how it works.

![YOLO in action](YOLO.jpg)

In this repo you'll find:

- **YOLO-CoreML:** A demo app that runs the YOLO neural network on Core ML.
- **Converter:** The scripts needed to convert the original DarkNet YOLO model to Core ML.

To run the app:

1. execute download.sh to download the pre-trained model
`% sh download.sh`
2. open the **xcodeproj** file in Xcode 9 and run it on a device with iOS 11 or better installed.

The reported "elapsed" time is how long it takes the YOLO neural net to process a single image. The FPS is the actual throughput achieved by the app.

> **NOTE:** Running these kinds of neural networks eats up a lot of battery power. The app can put a limit on the number of times per second it runs the neural net. You can change this in `setUpCamera()` by changing the line `videoCapture.fps = 50` to a smaller number.

## Converting the models

> **NOTE:** You don't need to convert the models yourself. Everything you need to run the demo apps is included in the Xcode projects already. 

If you're interested in how the conversion was done, check the [instructions](Converter/).

