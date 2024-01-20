
import os
import wget
import shutil
import subprocess

model_name = "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8"
model_url = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz"
model_path = "Tensorflow/pre_trained_model"

# Download the model
wget.download(model_url)

# Move the downloaded file to the specified model_path
shutil.move(f"{model_name}.tar.gz", model_path)

# Change the current working directory to model_path
subprocess.run(f"cd {model_path} && tar -zxvf {model_name}.tar.gz", shell=True)

# Remove the tar.gz file
os.remove(os.path.join(model_path, f"{model_name}.tar.gz"))