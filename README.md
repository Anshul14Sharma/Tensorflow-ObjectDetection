# Tensorflow-ObjectDetection
Object Detection (Computer Vision) on the Web with WebRTC and TensorFlow  

# About TensorFlow:  
TensorFlow is an open source software library machine learning framework for numerical computation which uses data flow graphs/computational graph as its computational model.  
Nodes in the graph represents mathematical operations, while the graph edges represent the multidimensional data arrays (known as tensors).  
The flexible architecture allows you to deploy computation to one or more CPUs or GPUs in a desktop, server, or mobile device with a single API.  
TensorFlow was originally developed by researchers and engineers working on the Google Brain Team within Google's Machine Intelligence research organization for the purposes of conducting machine learning and deep neural networks research, but the system is general enough to be applicable in a wide variety of other domains as well.  

Installing TensorFlow on Ubuntu:  
This module explains how to install TensorFlow on Ubuntu. Although these instructions might also work on other Linux variants.  
System Requirements:  
    • 64-bit desktops or laptops   
    • Ubuntu 14.04 or higher (Mine was Ubuntu 16.04 LTS)  
TensorFlow with CPU support only. If your system does not have a NVIDIA GPU, you must install this version. Note that this version of TensorFlow is typically much easier to install, if you have an NVIDIA GPU, we recommend installing GPU version.  

There are 4 ways through which you can install tensorflow on Ubuntu. The supported choices are as follows:
    • Virtualenv  
    • "native" pip  
    • Docker  
    • Anaconda  
Installing with native pip  
You may install TensorFlow through pip, choosing between a simple installation procedure or a more complex one.  
Note: The REQUIRED_PACKAGES section of setup.py lists the TensorFlow packages that pip will install or upgrade.  
Prerequisite: Python and Pip  
Python is automatically installed on Ubuntu. Take a moment to confirm that one of the following Python versions is already installed on your system:  
Command: python -v  
    • Python 2.7   
    • Python 3.4+  
The pip or pip3 package manager is usually installed on Ubuntu. Take a moment to confirm (by issuing a pip -V or pip3 -V command) that pip or pip3 is installed.  
Command: pip -V or pip3 -V  
We strongly recommend version 8.1 or higher of pip or pip3. If Version 8.1 or later is not installed, issue the following command, which will either install or upgrade to the latest pip version:  
$ sudo apt-get install python-pip python-dev   # for Python 2.7  
$ sudo apt-get install python3-pip python3-dev # for Python 3.n  
Install TensorFlow  
Assuming the prerequisite software is installed on your Linux host, take the following steps:  
    1. Install TensorFlow by invoking one of the following commands:  
       $ pip install tensorflow      # Python 2.7; CPU support (no GPU support)  
       $ pip3 install tensorflow     # Python 3.n; CPU support (no GPU support)  
       $ pip install tensorflow-gpu  # Python 2.7;  GPU support  
       $ pip3 install tensorflow-gpu # Python 3.n; GPU support  
       If the preceding command runs to completion, you should now validate your installation. 
    2. (Optional.) If Step 1 failed, install the latest version of TensorFlow by issuing a command of the following format: 
       $ sudo pip  install --upgrade tfBinaryURL   # Python 2.7  
       $ sudo pip3 install --upgrade tfBinaryURL   # Python 3.n  
       where tfBinaryURL identifies the URL of the TensorFlow Python package. The appropriate value of tfBinaryURL depends on the operating system, Python version, and GPU support. Find the appropriate value for tfBinaryURL here. For example, to install TensorFlow for Linux, Python 3.4, and CPU-only support, issue the following command:  
        $ sudo pip3 install --upgrade  
        https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.5.0-cp34-cp34m-linux_x86_64.whl  
If this step fails, see Common Installation Problems.
Next Steps 
After installing TensorFlow, validate your installation.  
Run a short TensorFlow program  
Invoke python from your shell as follows:  
$ python   
Enter the following short program inside the python interactive shell:  
# Python  
import tensorflow as tf  
hello = tf.constant('Hello, TensorFlow!')  
sess = tf.Session()  
print(sess.run(hello))  
If the system outputs the following, then you are ready to begin writing TensorFlow programs:  
Hello, TensorFlow!  
If you are new to TensorFlow, see Getting Started with TensorFlow.  
If the system outputs an error message instead of a greeting, see Common installation problems.  
Installation Dependencies    
Tensorflow Object Detection API depends on the following libraries:  
    • Protobuf 2.6   
    • Pillow 1.0  
    • lxml  
    • tf Slim (which is included in the "tensorflow/models/research/" checkout)   
    • Jupyter notebook  
    • Matplotlib  
    • Tensorflow  
For detailed steps to install Tensorflow, follow the Tensorflow installation instructions. A typical user can install Tensorflow using one of the following commands: 
# For CPU  
pip install tensorflow  
# For GPU  
pip install tensorflow-gpu  
The remaining libraries can be installed on Ubuntu 16.04 using via apt-get:  
sudo apt-get install protobuf-compiler python-pil python-lxml  
sudo pip install jupyter  
sudo pip install matplotlib  
Alternatively, users can install dependencies using pip:  
sudo pip install pillow  
sudo pip install lxml  
sudo pip install jupyter  
sudo pip install matplotlib  
Protobuf Compilation  
The Tensorflow Object Detection API uses Protobufs to configure model and training parameters. Before the framework can be used, the Protobuf libraries must be compiled. This should be done by running the following command from the tensorflow/models/research/ directory:  
# From tensorflow/models/research/  
protoc object_detection/protos/*.proto --python_out=.  
Add Libraries to PYTHONPATH  
When running locally, the tensorflow/models/research/ and slim directories should be appended to PYTHONPATH. This can be done by running the following from tensorflow/models/research/:  
# From tensorflow/models/research/  
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim  
Note: This command needs to run from every new terminal you start. If you wish to avoid running this manually, you can add it as a new line to the end of your ~/.bashrc file.  
Testing the Installation  
You can test that you have correctly installed the Tensorflow Object Detection  
API by running the following command:  
python object_detection/builders/model_builder_test.py  
Code Walkthrough  
Part 1 — Make sure Tensorflow works  
To make sure the TensorFlow Object Detection API works, let’s start with a tweaked version of the official the Object Detection Demo Jupyter Notebook. I saved this file as object_detection_tutorial.py.  
Object Detection API Output:  
 The Object Detection API outputs 4 objects:  
    1. classes - an array of object names  
    2. scores - an array of confidence scores  
    3. boxes-  locations of each object detected (xmin, xmax, ymin, ymax)  
    4. num - the total number of objects detected (whisch has a probability above the defined threshold)  
classes, scores, and boxes are all equal sized arrays that parallel to each other, so classes[n] corresponds to scores[n] and boxes[n]. 
The size of the arrays is equal to the number of objects detectected by the object detection API. 

