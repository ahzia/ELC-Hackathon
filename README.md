# AVA - Communicate with Deaf
## Live Link:
[heroku](https://pacific-refuge-24298.herokuapp.com/)

## Description

Deaf have their own language of communication (sign language), they can only understand this language and also they write and read Faster on signwriting. Deaf did not have access to most of the facilities as much as ordinary people, Over 5% of the world's population – or 466 million people – has disabling hearing loss (432 million adults and 34 million children). and The natural language of around 500,000 deaf people in the US and Canada is American Sign Language (ASL). For many people who have been profoundly deaf from a young age, signing is their first language so they learn to read and write English as a second language. As a result, many deaf people have below-average reading abilities for English text and prefer to communicate using sign language A comprehensive approach to the task of enabling humans who cannot sign to communicate with Deaf would clearly require the development of a general-purpose speech to sign language converter. This in turn requires the solution of the following problems:

Automatic speech to text conversion (speech Recognition). Automatic translation of English text into a suitable representation of sign language. Display of this representation as a sequence of Signs using computer graphics techniques. for now, our focus is on solving the second problem

the most commonly used sign representation is the international signwriting developed by the Sutton Movement

 1) It is aimed at enabling us to write every sign or signed sentence of every country;
 2) Because the writing is pictorial, natural shapes and movements of signs can be realistically shown;
 3) Face expressions and body movements can be depicted, too.
.

## Tech Stack
We have used the following
1) Flask
2) Python
3) React
4) Tensorflow

## Libraries and dependencies
The libraries that will be required are
1) flask
2) tensorflow
3) numpy
4) unicodedata
5) matplotlib
6) sklearn

## Challenges we ran into

1) Gathering the data set was very challenging, we spend a lot of time understanding signwriting, finding sources for gathering datasets, extracting sentences from these sources, changing their format (2D to 1D), and preparing data set for training.
2) hen creating the model and deploying it in flask we face a few small challenges but we figured how to solve them. ## Accomplishments that we're proud of We are proud that we developed a concept that can be used to develop fully functional applications and websites for the deaf. as a result, the deaf would be able to use their own language to communicate with ordinary people. 
## What we learned 
We learned the process of creating, deploying, and hosting a Tensorflow model, also we are now familiar with sign language and signwriting ;). 
## What's next for Signwriting
Easy to hear First of all, we will improve our data set, our plan is to create a large data set (more than 20000 sentences) and use it to create a fully functional "website+mobile app" machine translator to translate any English sentence to signwriting and vice versa. next, we will create a mobile application to help Deaf understand ordinary people thoughts, This, in turn, requires the solution of the following problems:
  1) Automatic speech to text conversion (speech Recognition).
  2) Automatic translation of English text into a suitable representation of sign language. - (using AVA ML model)
  3) Display of this representation as a sequence of Signs using computer graphics techniques. (a sign Avatar) The system will get the text/speech from a smartphone, convert it to        sign language using machine translation, and represent the signs using a 3D character.

## Demo
[![Watch the video](https://img.youtube.com/vi/zHn95iAsg2Y/maxresdefault.jpg)](https://www.youtube.com/embed/zHn95iAsg2Y)
