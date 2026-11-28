

<h3>PyTorch Neural Network Classification using Convolutional 2D Layers, ReLU Activation Functions, Pooling, Stochastic Gradient Descent (SGD), and Cross Entropy Loss</h3>
<p>
This program classifies eight composite chirp waveforms.  Each waveform has ten chirp signals of the following types: 'linear', 'quadratic', 'logarithmic', and 'hyperbolic'.
The different chirp types are randomly chosen and are used as the input to a Short-Time Fourier Transform (STFT) implemented with the FFT algorithm.  The sample rate of the 
waveforms is 64 smaples/sec producing the Nyquist frequency of 32 Hz.  There are 32 time-slices of the waveforms using a hop interval of 2 samples.  The 64 samples of 
each 64-sample block are windowed with the Hamming function.  The 32 frequency bins and 32 time slices represent the input image fed to the neural network.  The image is 
normalized by the maximum value of the image which produces a floating point number in [0,1].  The target or label is the specific wavform as an 
integer in (0, 7), since there are eight waveforms.  This is termed one-hot vector since only one of the eight outputs is one and the rest are zeros; for example,
a waveform 3 target would be [0,0,1,0,0,0,0,0].  The user can specify 1, 2, or 3 convolutional layers.  So the input samples consist of a 32x32 image of the STFT and the 
target integer.  The user can specifiy the number of epochs.  The batch size is fixed at one.  The user can also specify the learning rate and momentum hyperparameters.
After the network is trained, the Cross Entropy Loss per epoch is displayed.  The testing phase follows the training.  A different set of samples is presented to
the neural network.  The number of sample inputs for each class and the number of correct decisions is recorded.  The test results for each class and the totals 
are shown in a table.  The activation function used for each fully-connected layer is Rectified Linear Unit (ReLU).  The Cross Entropy Loss function is used to determine
the output error and backpropagation is used to distribute the error to previous-layer weights and biases using partial derivatives and the chain rule of 
Differential Calculus.  PyTorch handles all this transparently to the user with Tensors and Autograd.  The Cross Entropy Loss function uses Softmax to estimate
the probabilites of each output class and then takes the negative log of these values.  The maximum of these outputs is the predicted class.  Again this is 
done under the covers; the library handles all the details.
</p>
<p>
The architecture consists of the 32x32 input image, 1-3 2D convolutional layers using 3x3 and 5x5 kernels, ReLU, Max Pooling layers using 2x2 subsample, flattening the last layer of Feature Maps
to a linear array, a fully-connected Linear layer, and the 8x1 one-hot-vector output.  The three different size convolutional layers are illustrated in the diagrams below.
</p>
<p>
This program uses PyTorch (without a GPU or CUDA), Python 3.13, SciPy, Matplotlib, and NumPy.
</p>
<img width="896" height="254" alt="PytorchConv2DReluCrossEntropy" src="https://github.com/user-attachments/assets/13ba4ff4-5020-421b-89a7-439fcc02265a" />
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
