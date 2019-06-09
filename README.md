## Classification on cassava disease dataset of Kaggle


Your directory tree should look like this:

   ```
   ${ROOT}
   ├── cassava
   | ├── train
   | | ├── cbb
   | | ├── cbsd
   | | ├── cgm
   | | ├── cmd
   | | ├── healthy
   | ├── test
   | | ├── 0
   | ├── extraimages
   | | ├── 0
   ├── dataloaders
   ├── networks
   ├── utils
   ├── config.py
   ├── main.py
   └── README.md
   ```
   
You can:

	Train your model with inception v4 network using input image resolution 560, batch size 16 with:

	```
	python main.py --arch inceptionv4 --model_input_size 560 --batch_size 16
	```
	
	If you want to resume training from a checkpoint, you can use:

	```
	python main.py --arch inceptionv4 --model_input_size 560 --batch_size 16 --resume_path <path_to_pth_file>
	```
	
	Test your trained model from a checkpoint file using:
	
	```
	python main.py --arch inceptionv4 --model_input_size 560 --batch_size 16 --train False --test true --resume_path <path_to_pth_file>
	```
	
	Use validation by splitting training data using:
	
	```
	python main.py --arch inceptionv4 --model_input_size 560 --batch_size 16 --validate true --train_percentage 0.8
	```
	
	
	