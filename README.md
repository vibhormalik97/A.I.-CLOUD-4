# A.I.-CLOUD-4

Find the Report and screenshots in the file: Assignment4_Report.pdf

Our embedding dictionary can be accessed from: https://twittertextdata.s3.amazonaws.com/glove.twitter.27B.25d.txt

To successfully run the file:

Add the downloaded embeddings to AI_Cloud4/glove.twitter.27B.25d.txt 

Use the following command on terminal to run sentiment_training.py

python sentiment_training.py --train "training/" --eval "eval/" --dev "dev/" --num_epoch 1 --config "training_config.json" --model_output_dir "."
