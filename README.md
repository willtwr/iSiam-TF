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

# (OPTIONAL) 7. Install nvidia-ml-py for automatically selecting GPU
pip install nvidia-ml-py
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

## Benchmark
You can use the `run_SiamFC.py` in `benchmarks` directory to integrate with the OTB evaluation toolkit. The OTB toolkit has been ported to python. The original version is [here](https://github.com/jwlim/tracker_benchmark). However, you may want to use my [custom version](https://github.com/bilylee/tracker_benchmark) where several bugs are fixed. 

Assume that you have followed the steps in `Tracking` or `Training` section and now have a pretrained/trained-from-scratch model to evaluate. To integrate with the evaluation toolkit, 
```bash
# Let's follow this directory structure
# Your-Workspace-Directory
#         |- SiamFC-TensorFlow
#         |- tracker_benchmark
#         |- ...
# 0. Go to your workspace directory
cd /path/to/your/workspace

# 1. Download the OTB toolkit
git clone https://github.com/bilylee/tracker_benchmark.git

# 2. Modify line 22 and 25 in SiamFC-TensorFlow/benchmarks/run_SiamFC.py accordingly. 
# In Linux, you can simply run
sed -i "s+/path/to/SiamFC-TensorFlow+`realpath SiamFC-TensorFlow`+g" SiamFC-TensorFlow/benchmarks/run_SiamFC.py

# 3. Copy run_SiamFC.py to the evaluation toolkit
cp SiamFC-TensorFlow/benchmarks/run_SiamFC.py tracker_benchmark/scripts/bscripts

# 4. Add the tracker to the evaluation toolkit list
echo "\nfrom run_SiamFC import *" >> tracker_benchmark/scripts/bscripts/__init__.py

# 5. Create tracker directory in the evaluation toolkit
mkdir tracker_benchmark/trackers/SiamFC

# 6. Start evaluation (it will take some time to download test sequences).
echo "tb100" | python tracker_benchmark/run_trackers.py -t SiamFC -s tb100 -e OPE

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
