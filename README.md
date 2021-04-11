# Electropalatography-Audio Speech Enhancement


This model is written in pytorch 1.4.0. You first need to install some packages:
```
pip install -r requirements.txt
```
or
```
conda install --yes --file requirements.txt
```

## Training
To train the model, you need to specify the ``--dataset_path`` folders.
The dataset stores clean speech, noisy speech, and the EPG signals.
These sentences are recorded based on the Taiwan Mandarin hearing in noise test (Taiwan MHINT).

See [``main.py``](main.py) for more detailed training settings.

For the model structure.
You can freely construct a multimodel according to the specifications of [MultiModal_SE](model.py#L35) input parameters.

The main training process is shown in the ``Train EASE EF.ipynb`` file for early fusion model and ``Train EASE LF.ipynb`` file for late fusion model.

## Testing & Visualization
To test the model, [``Testing.ipynb``](Testing.ipynb) shows the performance and the spectrogram of some specific models in specific test sample.

[``analyze``](utils.py#316) can calculate the average of PESQ, STOI and ESTOI. And save the results in the Evaluation folder.

[``avg_analyze``](utils.py#502) can show the average performance of the selected models on the specific metrics in a bar chart.
