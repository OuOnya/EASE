# EPG2S: Speech Generation Based on Electropalatography Signals Through Multimodal Learning


This model is written in pytorch 1.4.0. You first need to install some packages:
```
pip install -r requirements.txt
```
or
```
conda install --yes --file requirements.txt
```

Or you can create the environment from the environment.yml file:
```
conda env create -f environment.yml
```

## Training
To train the model, you need to specify the ``--dataset_path`` folders.
The dataset stores clean speech, noisy speech, and the EPG signals.
These sentences are recorded based on the Taiwan Mandarin hearing in noise test (Taiwan MHINT).

See [``main.py``](main.py) for more detailed training settings.

For the model structure.
You can freely construct a multimodel according to the specifications of [MultiModal_SE](model.py#L38) input parameters.

The main model building and training process is shown in these ipynb files:
1. [``Train_EPG2S_baseline``](Train_EPG2S_baseline.ipynb) file for baseline model.
2. [``Train_EPG2S_EF``](Train_EPG2S_EF.ipynb) file for early fusion model.
3. [``Train_EPG2S_LF``](Train_EPG2S_LF.ipynb) file for late fusion model.
4. [``Train_EPG2S_EPG``](Train_EPG2S_EPG.ipynb) file for EPG-only baseline model.


## Testing & Visualization
To test the model, use [``Test_Results.ipynb``](Test_Results.ipynb) file. It can also compare the results of different models.
There are two comparisons for discussion::
1. Use audio and EPG signals as input
2. Only use the EPG signal as input

[``analyze``](utils.py#L394) function can calculate the average of PESQ, STOI and ESTOI. And save the results in the Evaluation folder.

[``avg_analyze``](utils.py#L595) function can show the average performance of the selected models on the specific metrics in a bar chart.
