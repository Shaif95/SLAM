The code in tensorflow creates binary files for the given images after training. The binary codes are saved in a txt file.
The keras code can be used to train on a large dataset. The saved model can be used to create binary codes and search realtime.
The Network can be trained with the dataset( 2 square size required) placed in 'images' folder. Right  now it would resize the image in 32*32 and start training.
Larger images would produce better results. That would need changes to data and def files. 

Searching is given in a Notebook.
The binary code gives a lot of false positive results.
To deal with that, it searches for 10 nearest images. If there are three or more consecutive images then the loop is considered closed. 
This is just for eliminating false positives. 

