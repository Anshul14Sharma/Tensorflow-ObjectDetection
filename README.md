# Tensorflow-ObjectDetection   
First go with the installation of TensorFlow.  
Go to the folder  and run using following command:-  
    python3 penDetection.py
# Steps for creating own dataset :-   
-> Tensorflow object detection API uses the TFRecord file format, so we need to convert our dataset to this file format.  
-> To prepare the input file for the API you need to consider things. Firstly, you need an RGB image which is encoded as jpeg or png and secondly you need a list of bounding boxes (xmin, ymin, xmax, ymax) for the image and the class of the object i the bounding box. I have created only one class (i.e. Pen). For one class it was easy as we go for more number of classes it becomes time comsuming.  
-> I scraped 156 images of pen (mainly jpg) from Google images, where I made sure that the images have large variation and scale, pose and lighting.  
-> After that I hand labeled them manually using [LabelImg](https://www.github.com/tzutalin/labelImg). LabelImg is a graphical image annotation tool. It is written in Python and uses for its graphical interface. It's easy to use and annotations are saved as XML files in the PASCAL VOC format which means that I could use create_pascal_tf_record.py script.  
-> Finally, after labeling the images I refer to the script that converted the XML file to csv and then to TFRecords. 

# Steps to train the created dataset :-  
-> 
