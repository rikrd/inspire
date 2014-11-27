# INSPIRE Challenge Development Kit 
This development kit is a set of tools to help participating in the INSPIRE Challenge.


## INSPIRE Challenge

The main goal of the INSPIRE Challenge is to evaluate models of microscopic intelligibility in the case of single 
word listening.  
The term microscopic refers to the level of detail of the intelligibility description.  
In contrast to previous works we focus on intelligibility at a phoneme level.  

Our objective is to evaluate the ability of a model to predict the confusions produced by certain stimuli on a 
set of listeners.

### Dataset

For this challenge we propose the use of the data collected running the BigListen experiments.  

This experiment consists of presenting to listeners a token composed of a single word utterance mixed with noise.  
The task of the listener is to report the word corresponding to the utterance. Tokens that consistently produce word 
confusions on multiple listeners are selected. Tokens for which the word is correctly identified are discarded.  

Different types of noises have been used.  
  *  **SSN** : Speech Shaped Noise  
  *  **BMNn** : n-speaker Babble Modulated Noise  
  *  **BABn** : n-speaker Babble Noise  

The noises are mixed at different SNR levels.  

### Tasks



## INSPIRE Development Kit

The kit consists of a set of Python scripts to build, test and evaluate baseline models to participate in the INSPIRE 
Challenge.  
The kit also contains scripts to generate useful intermediate data such as pronunciation dictionaries, 
forced alignments and recognitions.  

The main goal of the kit is provide an ease-of-use for most common tasks with a focus on flexibility for most common 
customizations such as the use of different audio features or lexicons.  


### Prerequirements

We here introduce the needed requirements to successfully run the scripts in the kit.

#### Terminal encoding
Since some of the datasets are in foreign languages and non-ASCII characters may be used in lexicons, label files and
filenames we must take special care with the encoding of all textual information.  

Set your locale to a UTF-8 encoding.  You can do this by setting your locale envorinoment variable in your bashrc:

```bash
echo "export LC_ALL=en_US.UTF-8" > $HOME/.bashrc
source $HOME/.bashrc
```

Additionally make sure to download the datasets provided by us.  The filenames in these datasets are encoded in 
UTF8 NFD (Normalized Form Decomposed) of filenames which allow to work with the same scripts on GNU/Linux, 
Windows and Mac. This encoding has been chosen because the default filesystem in Mac (HFS+) forces this encoding, while
most other filesystems default to NFC (Normalized Form Composed) but allow both.

#### Python dependencies
The scripts are written for Python 2.x

The Python libraries required are:
  * numpy
  * scipy
  * 


#### HTK
The scripts in this kit assumes you have HTK installed and the commands are in the PATH.

### Tools




### Use cases

#### Generate lexicons (pronunciation dictionaries)

```bash
./generate_dictionary.py -d data/spanish_dictionary.txt -c ~/data/Spanish\ confusions/conf_11_03_13_final.csv -w data/spanish_wordlist.txt -o data/spanish_full_ipa.dict -f ipa -v es
./generate_dictionary.py -d data/spanish_dictionary.txt -c ~/data/Spanish\ confusions/conf_11_03_13_final.csv -w data/spanish_wordlist.txt -o data/spanish_full_xsampa.dict -f xsampa -v es
./generate_dictionary.py -d data/spanish_dictionary.txt -c ~/data/Spanish\ confusions/conf_11_03_13_final.csv -w data/spanish_wordlist.txt -o data/spanish_full_htk.dict -f xsampa -v es -e
```

#### Preparing for training

```bash
./prepare.py -T 0 -a /Users/rmarxer/data/INSPIRE/EN_wordset -d data/english_dictionary.txt -n "{1: 1000, 2:500}" -r 1 -o preparation/english_s1s2
```


#### Training a speech recognition model

```bash
./train.py -T 0 -p data/english_phones_class.hed.orig -i preparation/english_s1s2 -o models/english_s1s2
```

#### Using different features

```bash
./train_reestimate.py -T 0 -i models/english_s1s2 -o models/english_s1s2_ratemap -x ~/dev/python_ratemap/ratemap.py -xp '--cuberoot -S'
```

#### Adapting a speech recognition model

```bash
./adapt_cmllr.py -T 1 -n "{3: 1000, 4:500}" -r 1 -i models/english_s1s2
```

#### Testing a speech recognition model

```bash
./test.py -T 1 -n "{1: 1000, 2:500}" -r 1 -i models/english_s1s2
```



## Tutorial for participation
In this tutorial we will create a set of baseline models to perform all the tasks of the challenge.
We will make use of the tools provided by the Development kit, and for each of them we will evaluate the performance acheived.

### Task 1: Where are the confusions?

This task consists in predicting where in the word the confused phonemes will be.
We will start from a random guess and increasingly add knowledge to our model to make better predictions:

 * Random guess: random subset from all the phonemes of the presented word

### Task 2: What are the confusions?

This task consists in predicting what are the confusions that have occured at given locations of the presented word.

 * Random guess: a random guess of what kind of operation and the phonemes involved (in the case it is a insertion or replacement)
    

### Task 3: Full confusion prediction

This task consists in predicting the responses from the listeners to the stimuli presented.
In this case as a baseline we simply present an ASR-based predictor implemented as an N-Best recogniser.

 * Random guess: a random guess of how often the reported word will match the spoken word

