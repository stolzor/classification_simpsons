<h1>Introduction</h1>
Hello everyone. At first, I trained ResNext50_32x4d to solve this problem, but I decided to train RegNet800. The motivation was to compare these two architectures and try the power of the network design space in practice . And as a result, RegNet shows itself better, so I'll leave the weights of this model as an estimate.
<h2>Setting up the environment</h2>
You can download all the necessary libraries, frameworks using requirements.txt

````
pip install -r requirements.txt
````

<h2>Train</h2>
Before launching train.py , run data.py . Since in this task I am doing data augmentation (because there are few of them).

````
data.py -> train.py
````

<h2>Eval</h2>
If you want to use the Simpsons classification model, then run the eval script.py and specify the path to your picture. And then the script will give you who is depicted in your picture.

````
launch eval.py -> enter path image
````