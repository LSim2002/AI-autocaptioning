# AI-autocaptioning


Please note that this code was written quite a long time ago, and if I were to redo it today I would probably change most of the structure/code/models used. 

The idea was to make photo captioning as fast as possible, as the press is very comptetitive when it comes to the speed at which images are taken, labeled/captioned and then sold.  

The different models used could find people on an image, run facial recognition to identify them (they have to be celebrities, or at least have their face in the database to be recognized), estimate their pose (handshaking / walking / running).


So, with only the image, and given the place and time of photography, and no other information, my script returns a caption such as :



![This should be an image of Bolt running...](https://st4.depositphotos.com/21607914/24104/i/450/depositphotos_241041352-stock-photo-usain-bolt-jamaica-crosses-finish.jpg "BoltRunning")
<p align="center"><strong><em>"Ussain bolt showcases his athletic prowess at the Adidas Grand Prix, an annual track and field event, as he sprints towards victory in New York City, USA."</em></strong></p>

Or 

![This should be an image of Putin and Obama handshaking...](https://s.abcnews.com/images/International/ap_obama_putin_handshake_float_jc_151201_16x9_1600.jpg "PutinAndObama")

<p align="center"><strong><em>"Putin and Obama engage in a warm handshake during the Paris Summit, reflecting the high level of interaction among world leaders discussing climate change measures in November 2015."</em></strong></p>



The models used are:

* Google's pose detector (MediaPipe) to get the coordinates of the body landmarks
* A dense neural network (TensorFlow) trained on very few examples (~100 examples per class), yet surprisingly effective, that takes as input the body joints angles and ouputs a pose (handshaking / walking / running).
* A facial embeddings generator (MediaPipe) used for facial recognition (euclidean distance).
* A Natural Language Processing model (ChatGPT 4.0, used with OpenAI API) that, with the names of the people on the image, their pose, the place and the date, coupled with ingenious prompt engineering, generates a caption in english words.   