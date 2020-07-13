
# DeepSearch Code And Data

This folder contains the files used for evaluating DeepSearch and other competing techniques. Due to github filesize restrictions, some large files containing the models to be attacked have been deleted. These can be found along with the entire code at this link:https://figshare.com/s/2faecff053c89b00e598

The Top Level Directory Contains 5 Sub Directories, which are detailed hereafter. Curly braces are used to indicate multiple similar files, with each element inside curly braces indicate the partial name for one such file. Inside Curly braces -(hyphen) represents the empty string.

## ImageNet

This contains code to run various attacks on the ImageNet Validation Dataset. The Files in this folder are:

*  **images/**: This directory contains the 1000 images used to test the performance of all the attacks. These belong to the validation set from the ImageNet Large Scale Visual Recognition Challenge 2012 and have been chosen so ass to contain 1 correctly classified image from each dataset

*  **imgntWrapper.py**: This is the Wrapper file for the ImageNet tests. It loads the inception v3 model provided by pytorch and provides a simple interface over it, and also loads and preprocesses the images to be attacked. This file is used by all attack files

*  **LazierGreedy.py, Bandits.py, QLNES.py**: These files contain the implementations for the five attacks discussed in our paper, as well as some other methods that we tried out in the course of our research.

*  **test{Bandits/DS/Parsi/QLNES/Simba}.py**: These files are the actual executables that runs its attack(Bandits TD, DeepSearch, Parsimonious, Query Limited NES and SimBA in Pixel Space respectively) and stores its result. All files can be executed as should necessary prerequisites be satisfied, all files can be run using `python filename`. testParsi.py and testDS.py take two optional arguments, the loss function to be used(logit(default) or xent) and the initial group size(some power of 2, default=32), the second of which is useful for sensitivity experiments. On execution, this will create a directory named after the timestamp for the instant at which it was created containing the following:

	* A pickle file *data.pkl* containing a python dictionary whose keys are the index of the image and the values a pairs indicating whether or not the attack succeeded on the said image and how many queries it took

	* A text file *log.txt* containing the output from the attack

	* A series of files containing the attacked images(if the attack failed on an image then the corresponding file is irrelevant). Bandits TD and QUery Limited NES attacks save images as batches, while the rest save them as individual files.

*  **testRef.py**: This file is to test the Refinement process. It is mostly identical to the attack files. However, this can run only after *testDS.py* has been run and its output stored. After doing that, the variables `dir` and `dic` need to be manually set to the output directory and the *data.pkl* file from the said attack. Also, if *testDS.py* is run with arguments, the exact same arguments need to be used to run this file.The output is similar to that of the other attack files, except that the *data.pkl* file now has list of query-distace tuples(after each stage of refinement) as the value.

  

## SVHN and CIFAR

These two directories contain code to run various attacks on the respective Validation Datasets. The Files in this folder are mostly identical, and hence they are documented together:

* **model_{un/-}defended/**: These folders contain the TensorFlow checkpoints for the models to be attacked.
* **model.py**: This file builds the Neural network in tensorflow. it has been borrwed as-is from the Madry CIFAR Challenge.
* **{def_/-}indices.pkl**: These pickle files contain a list of indexes 1000 randomly selected images that are correctly classified by the respective models. The "def" suffix indicates that this file contains indices for the defended network.  
* **madry{SVHN/CIFAR}{Undef/-}Wrapper.py** :These are the Wrapper files for the respective tests. They load the model checkpoint and provides a simple interface over it(identical to the one for ImageNet, allowing for some abstraction), and also loads and preprocesses the images to be attacked. Note that unlike ImageNet, the entire test set is loaded, and the respective indices file needs to be used for selecting the right set of images These files are used by all attack files for a given model.
* **test.mat**: This file is only present in the SVHN directory and contains the testSet for SVHN. It has been downloaded from [here](http://ufldl.stanford.edu/housenumbers/)
*  **LazierGreedy.py, Bandits.py, QLNES.py**: These files contain the implementations for the five attacks discussed in our paper, as well as some other methods that we tried out in the course of our research.

*  **test{Bandits/DS/Parsi/QLNES/Simba}.py**: These files are the actual executables that runs its attack(Bandits TD, DeepSearch, Parsimonious, Query Limited NES and SimBA in Pixel Space respectively) and stores its result. All files can be executed as should necessary prerequisites be satisfied, all files can be run using `python filename <defense>`, where `defence` is "def" for the defended network and "undef" for the undefended network. testParsi.py and testDS.py take two more optional arguments, the loss function to be used(logit(default) or xent) and the initial group size(some power of 2, default=4), the second of which is useful for sensitivity experiments. On execution, this will create a directory named after the timestamp for the instant at which it was created containing the following:

	* A pickle file *data.pkl* containing a python dictionary whose keys are the index of the image and the values a pairs indicating whether or not the attack succeeded on the said image and how many queries it took

	* A text file *log.txt* containing the output from the attack

	* A series of files containing the attacked images(if the attack failed on an image then the corresponding file is irrelevant). Bandits TD and QUery Limited NES attacks save images as batches, while the rest save them as individual files.

*  **testRef.py**: This file is to test the Refinement process. It is mostly identical to the attack files. However, this can run only after *testDS.py* has been run and its output stored. After doing that, the variables `dir` and `dic` need to be manually set to the output directory and the *data.pkl* file from the said attack. Also, if *testDS.py* is run with optional arguments, the exact same arguments need to be used to run this file. The output is similar to that of the other attack files, except that the *data.pkl* file now has list of query-distace tuples(after each stage of refinement) as the value.

## PreEvaluatedData
This folder contains the *data.pkl* files from attacks that were run by the authors. It contains 3 subfolders:
* **UndefData/**: Contains the files from attacks over the undefended SVHN and CIFAR network. The prefix indicates which dataset a file came from.
* **DefData/**: Contains the files from attacks over the defended SVHN and CIFAR network. The prefix indicates which dataset a file came from.
* **ImagenetData/**: Contains the files from attacks over ImageNet. These files have no prefix.

Each of these folders contains 1 file per attack. The file names are as follows:
* DS_XE:DeepSearcch with cross entropy loss.
* DS_CW:DeepSearch with logit loss
* DR:DeepSearch with Refinement and appropriate loss. In the imagenet folder, both losses are provided
* PS_XE:Parsimonious BlackBox Attack with cross entropy loss
* PS_CW:Parsimonious BlackBox Attack with logit loss
* Bandits: Bandits TD Attack
* Simba: Simple Blackbox Attack in pixel space
* QLNES: Query limited NES attack

Apart from these files, there are two subdirectories in each of these directories, which contain the files from the sensitivity experiments. For these files, the suffix indicates the initial group size

## Examples
This folder contains 4 folders for the Defended and Undefended versions of CIFAR10 and SVHNC, each of which contains 10 examples(1 from each class) of images attacked by our method.  
