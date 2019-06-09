## Visual classification on cassava disease dataset of Kaggle

This is the implementation of Cassava Disease Fine-Grained Visual Classification Challenge, 5th place entry on Kaggle.
Link to competition: https://www.kaggle.com/c/cassava-disease
Networks used in this repository are PyTorch official implementations or from https://github.com/Cadene/pretrained-models.pytorch, with small alterations.

Requires pytorch >= v1.0.0

Download cassava disease dataset from https://www.kaggle.com/c/cassava-disease/data and put it into the root directory ${ROOT}

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
   
### Training and Testing

Train your model with inception v4 network using input image resolution 560, batch size 16 with:

	python main.py --arch inceptionv4 --model_input_size 560 --batch_size 16
	
	---o---
	
If you want to resume training from a checkpoint, you can use:

	python main.py --arch inceptionv4 --model_input_size 560 --batch_size 16 --resume_path <path_to_pth_file>
	
	---o---
	
Test your trained model from a checkpoint file using:
	
	python main.py --arch inceptionv4 --model_input_size 560 --batch_size 16 --train False --test true --resume_path <path_to_pth_file>
	
	---o---
	
Use validation by splitting training data using:
	
	python main.py --arch inceptionv4 --model_input_size 560 --batch_size 16 --validate true --train_percentage 0.8
	
	---o---
	
	
	