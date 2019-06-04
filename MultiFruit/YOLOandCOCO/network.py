from imageai.Detection import ObjectDetection
import os
from os import listdir
from os.path import isfile, join
from PIL import Image

inpath = "verify"
outpath = "verify/out"
images = [f for f in listdir(inpath) if isfile(join(inpath, f))]

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()

# load coco dataset weights
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5")) 
detector.loadModel()

# restrict labels in coco to only fruits
custom_objects = detector.CustomObjects(apple=True, banana=True, orange=True) 

# process images in inpath to outpath
for i in images:
    print("Doing: "+i)
    idir = inpath + "/" + i
    odir = outpath + "/" + i
    detections = detector.detectCustomObjectsFromImage(custom_objects=custom_objects, input_image=os.path.join(execution_path , idir), output_image_path=os.path.join(execution_path , odir))
    
    for eachObject in detections:
        print(eachObject["name"] , " : " , eachObject["percentage_probability"] )