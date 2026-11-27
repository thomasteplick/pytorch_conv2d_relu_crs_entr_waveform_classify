<pre>
C:\Users\Thoma\eclipse-workspace\PyTorchWaveformClassifyReLUCrsEntr\Classify>conv2DReLUcrsEntr.py -h
usage: conv2DReLUcrsEntr [-h] {100,200,500,1000,2000,5000,10000} {1,2,10,16,32,64} {0.001,0.002,0.01,0.02} {0.5,0.6,0.7,0.8,0.9} {1,2,3} {5,10,15,20}

Convolution Neural Network (ReLU, Cross Entropy Loss) for waveform classification using PyTorch

positional arguments:
  {100,200,500,1000,2000,5000,10000}
                        the number of epochs for training and testing
  {1,2,10,16,32,64}     the batch size for training and testing
  {0.001,0.002,0.01,0.02}
                        learning rate hyperparameter for training
  {0.5,0.6,0.7,0.8,0.9}
                        momentum hyperparameter for training
  {1,2,3}               number of convolution 2D layers
  {5,10,15,20}          signal-to-noise ratio using gaussian noise

options:
  -h, --help            show this help message and exit
</pre>
<h4>Waveform Time Domain Plots, 10 chirps, 64 samples/sec</h4>
<img width="899" height="1020" alt="image" src="https://github.com/user-attachments/assets/60792f03-dc0c-4251-9089-c26d206bd2a8" />
<h4>Waveform Short-Time Fourier Transform (STFT), 64 samples/sec</h4>
<img width="894" height="1017" alt="image" src="https://github.com/user-attachments/assets/f3df8405-26e5-4366-a6e7-ad55f86b26f3" />
<pre>
C:\Users\Thoma\eclipse-workspace\PyTorchWaveformClassifyReLUCrsEntr\Classify>conv2DReLUcrsEntr.py 100 1 .001 .5 2 10
Using cpu device
Model: NeuralNetwork(
  (conv2D_relu_crsEntr_stack): Sequential(
    (0): Conv2d(1, 4, kernel_size=(5, 5), stride=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(4, 8, kernel_size=(3, 3), stride=(1, 1))
    (4): ReLU()
    (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Flatten(start_dim=0, end_dim=-1)
    (7): Linear(in_features=288, out_features=8, bias=True)
  )
)
Display the waveform's STFT (spectrogram), no additive white Gaussian noise
Begin training the model: 2026-11-26 16:04:27.569000
**************************************************
**************************************************

Done training: 2026-11-26 16:04:59.680282
Elapsed time: 0:00:32.111282
Display the Cross Entropy Loss
Saved PyTorch Model State to data\conv2DReLUcrossEntropy.pth
Begin testing the model: 2026-11-26 16:08:12.012369

Done testing.  2026-11-26 16:08:12.075367
Elapsed time: 0:00:00.062998 Test Results:
|--------|--------|--------|--------|
|Class   |Count   |Correct |%Correct|
|--------|--------|--------|--------|
|1       |15      |15      |100     |
|--------|--------|--------|--------|
|2       |11      |11      |100     |
|--------|--------|--------|--------|
|3       |11      |11      |100     |
|--------|--------|--------|--------|
|4       |15      |15      |100     |
|--------|--------|--------|--------|
|5       |18      |18      |100     |
|--------|--------|--------|--------|
|6       |10      |10      |100     |
|--------|--------|--------|--------|
|7       |11      |11      |100     |
|--------|--------|--------|--------|
|8       |9       |9       |100     |
|--------|--------|--------|--------|
|Totals  |100     |100     |100     |
|--------|--------|--------|--------|  
</pre>
<h4>Lerning Curve, Two Convolutional 2D layers, SNR=10dB, Two Pooling layers, One fully-connected Linear Layer</h4>
<img width="635" height="551" alt="image" src="https://github.com/user-attachments/assets/d24ea248-85de-4a22-bc08-990ee0c09c26" />
<pre>
C:\Users\Thoma\eclipse-workspace\PyTorchWaveformClassifyReLUCrsEntr\Classify>conv2DReLUcrsEntr.py 100 1 .001 .5 3 10
Using cpu device
Model: NeuralNetwork(
  (conv2D_relu_crsEntr_stack): Sequential(
    (0): Conv2d(1, 4, kernel_size=(5, 5), stride=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(4, 8, kernel_size=(3, 3), stride=(1, 1))
    (4): ReLU()
    (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1))
    (7): ReLU()
    (8): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (9): Flatten(start_dim=0, end_dim=-1)
    (10): Linear(in_features=64, out_features=8, bias=True)
  )
)
Display the waveform's STFT (spectrogram), no additive white Gaussian noise
Begin training the model: 2026-11-26 21:28:40.178754
**************************************************
**************************************************

Done training: 2026-11-26 21:29:03.797108
Elapsed time: 0:00:23.618354
Display the Cross Entropy Loss
Saved PyTorch Model State to data\conv2DReLUcrossEntropy.pth
Begin testing the model: 2026-11-26 21:35:40.323884

Done testing.  2026-11-26 21:35:40.377455
Elapsed time: 0:00:00.053571 Test Results:
|--------|--------|--------|--------|
|Class   |Count   |Correct |%Correct|
|--------|--------|--------|--------|
|1       |10      |10      |100     |
|--------|--------|--------|--------|
|2       |11      |11      |100     |
|--------|--------|--------|--------|
|3       |12      |12      |100     |
|--------|--------|--------|--------|
|4       |10      |10      |100     |
|--------|--------|--------|--------|
|5       |17      |17      |100     |
|--------|--------|--------|--------|
|6       |12      |12      |100     |
|--------|--------|--------|--------|
|7       |14      |14      |100     |
|--------|--------|--------|--------|
|8       |14      |14      |100     |
|--------|--------|--------|--------|
|Totals  |100     |100     |100     |
|--------|--------|--------|--------|
</pre>
<h4>Learning Curve, Three Convolutional 2D layers, SNR=10dB, Three Pooling layers, One fully-connected Linear Layer</h4>
<img width="636" height="548" alt="image" src="https://github.com/user-attachments/assets/415d221e-7918-43ef-983d-265d1d140496" />
