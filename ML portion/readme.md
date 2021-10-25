## What kind of ML is being used here?
Natural language processing is used in order to convert the english characters to Sutton Sign Writing.

## About the approach
Here we have used attention model in order to facilitate the Seq2Seq learning. There is an encoder and decoder where the encoder receives the English text and the decoder performs the conversion from the latent space to Sign writing. Attention is where we emphasise more on certain parts of the sequence which will impact the overall output.
![image](https://www.tensorflow.org/images/seq2seq/attention_mechanism.jpg)

## Tools Used
For the implementation we have used Tensorflow 2.0 . Along with that we have other supporting libraries such as numpy, sklearn, unicodedata, matplotlib, re and time.

## About the Dataset
The dataset is the Sutton Sign Writing Dataset where the text file is tab seperated to differentiate the sign writing and English text. The dataset used for the problem statement is small in size but however needs to be improved and developed in the future. The documentation is found in the below link.

https://github.com/sutton-signwriting



