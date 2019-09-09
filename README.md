# iSiam-TF (under construction)
A TensorFlow implementation of the i-Siam tracker.

The codes were fetched and modified from https://github.com/bilylee/SiamFC-TensorFlow.

## Introduction

This is a TensorFlow implementation of [i-Siam: Improving Siamese Tracker with Distractors Suppression and Long-Term Strategies](na). 

## Prerequisite
The main requirements can be installed by:
```bash
# (OPTIONAL) 0. It is highly recommended to create a virtualenv or conda environment
conda create -n pytf python=2.7
source activate pytf

# 1. Install TensorFlow
pip install tensorflow    # For CPU
pip install tensorflow-gpu  # For GPU

# 2. Install scipy for loading mat files
pip install scipy

# 3. Install sacred for experiments logging
pip install sacred

# 4. Install matplotlib for visualizing tracking results
pip install matplotlib

# 5. Install opencv for preprocessing training examples
pip install opencv-python

# 6. Install pillow for some image-related operations
pip install pillow

# 7. Install nvidia-ml-py for automatically selecting GPU
pip install nvidia-ml-py

# 8. Follow instructions in http://got-10k.aitestunion.com/ to install their toolkits. 
```

## Training
```bash
# 1. Download and unzip the GOT-10k dataset http://got-10k.aitestunion.com/
# Now, we assume it is unzipped to /path/to/got10k
DATASET=/path/to/got10k

# 2. Clone this repository to your disk 
# (Skip this step if you have already done)
git clone https://github.com/willtwr/iSiam-TF.git

# 3. Change working directory
cd iSiam-TF

# 4. Create a soft link pointing to the GOT-10k dataset
mkdir -p data
ln -s $DATASET data/got10k

# 5. Prepare training data
python scripts/preprocess_got10k_data.py

# 6. Split train/val dataset and store corresponding image paths
python scripts/build_got10k_imdb_reg.py

# 7. Start training
python experiments/iSiam.py

# 8. (OPTIONAL) View the training progress in TensorBoard
# Open a new terminal session and cd to iSiam-TF, then
tensorboard --logdir=Logs/SiamFC/track_model_checkpoints/iSiam
```

## Benchmark OTB-100
Benchmark for OTB-100 uses the [custom OTB evaluation toolkit](https://github.com/bilylee/tracker_benchmark) where several bugs are fixed. 

```bash
# Assume directory structure:
# Your-Workspace-Directory
#         |- iSiam-TF
#         |- tracker_benchmark
#         |- ...
# 0. Go to your workspace directory
cd /path/to/Your-Workspace-Directory

# 1. Download the OTB toolkit
git clone https://github.com/bilylee/tracker_benchmark.git

# 2. Modify iSiam-TF/benchmarks/run_iSiam_otb.py if needed. 

# 3. Copy run_iSiam_otb.py to the evaluation toolkit
cp iSiam-TF/benchmarks/run_iSiam_otb.py tracker_benchmark/scripts/bscripts

# 4. Add the tracker to the evaluation toolkit list
echo "\nfrom run_iSiam_otb import *" >> tracker_benchmark/scripts/bscripts/__init__.py

# 5. Create tracker directory in the evaluation toolkit
mkdir tracker_benchmark/trackers/iSiam_otb

# 6. Start evaluation (it will take some time to download test sequences).
echo "tb100" | python tracker_benchmark/run_trackers.py -t iSiam_otb -s tb100 -e OPE

# 7. Get the AUC score
sed -i "s+tb50+tb100+g" tracker_benchmark/draw_graph.py
python tracker_benchmark/draw_graph.py
```

## Tracking
```bash
# 1. Clone this repository to your disk
git clone https://github.com/bilylee/SiamFC-TensorFlow.git

# 2. Change working directory
cd SiamFC-TensorFlow

# 3. Download pretrained models and one test sequence 
python scripts/download_assets.py

# 4. Convert pretrained MatConvNet model into TensorFlow format.
# Note we use SiamFC-3s-color-pretrained as one example. You
# Can also use SiamFC-3s-gray-pretrained. 
python experiments/SiamFC-3s-color-pretrained.py

# 5. Run tracking on the test sequence with the converted model
python scripts/run_tracking.py

# 6. Show tracking results
# You can press Enter to toggle between play and pause, and drag the 
# scrolling bar in the figure. For more details, see utils/videofig.py
python scripts/show_tracking.py
```

## License
iSiam-TF is released under the MIT License (refer to the LICENSE file for details).
