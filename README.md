# Tensorflow-ObjectDetection   
Install Tensorflow and dependencies:-  
```sudo pip3 install -r requirements.txt```  
First go with the installation of TensorFlow.  
Go to the folder object_detection and run using following command:-  
```python3 penDetection.py```
# Steps for creating own dataset :-   
-> Tensorflow object detection API uses the TFRecord file format, so we need to convert our dataset to this file format.  
-> To prepare the input file for the API you need to consider things. Firstly, you need an RGB image which is encoded as jpeg or png and secondly you need a list of bounding boxes (xmin, ymin, xmax, ymax) for the image and the class of the object i the bounding box. I have created only one class (i.e. Pen). For one class it was easy as we go for more number of classes it becomes time comsuming.  
-> I scraped 156 images of pen (mainly jpg) from Google images, where I made sure that the images have large variation and scale, pose and lighting.  
-> After that I hand labeled them manually using [LabelImg](https://www.github.com/tzutalin/labelImg). LabelImg is a graphical image annotation tool. It is written in Python and uses for its graphical interface. It's easy to use and annotations are saved as XML files in the PASCAL VOC format which means that I could use create_pascal_tf_record.py script.  
-> Finally, after labeling the images I refer to the script(xml_to_csv.py) that converted the XML to csv file and then (generate_tfrecord.py) to generate TFRecords.
# Steps to train the created dataset :-  
-> Place the created train.record and test.record in data directory of object detection.  
-> Define the class of object you want to detect .pbtxt file in training directory. In my case it is pen_label_map.pbtxt and made the following changes in the config file of model(ssd_mobilenet_v1.config)  
```num_classes: 1 (Number of classes we are detecting)  
batch_size: 10 (Adjust accordingly, because more value for batch can generate out-of-memory errors)  
fine_tune_checkpoint: "model_name/model.ckpt"  
num_steps:4000 (Number of steps required in training process)  
train_input_reader:{  
 	tf_record_input_reader {  
	input_path: "data/train.record"  
	}  
      	label_map_path: "data/pen_label_map.pbtxt"  
eval_input_reader: {  
 	tf_record_input_reader {  
	input_path: "data/test.record"  
	}  
```
-> save config file.  
->Use the following command to start the training process:-  
```python3 object_detection/train.py --logtostderr --train_dir=data/ --pipeline_config_path=data/ssd_mobilenet_v1_pen.config```  

--> Before we start experimenting with our newly trained model, we have to export the graph for inference. You can use the latest ckpt # from
your data directory.  
```python3 object_detection/export_inference_graph.py \  
   --input_type image_tensor \  
   --pipeline_config_path data/ssd_mobilenet_v1_pen.config \  
   --trained_checkpoint_prefic data/model.ckpt-# (last checkpoint id) \  
   --output_directory object_detection_graph  
```
After this, we can start detecting the images.


