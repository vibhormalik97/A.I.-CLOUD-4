# A.I.-CLOUD-4

The focus of this assignment was on building a pipeline for cleaning the text and training a CNN model for sentiment analysis.

The following AWS services were used:
- AWS Glue
- AWS Lambda
- AWS SageMaker

Find the Report and screenshots in the file: Assignment4_Report.pdf

Our embedding dictionary can be accessed from: https://twittertextdata.s3.amazonaws.com/glove.twitter.27B.25d.txt

To successfully run the file:

Add the downloaded embeddings to AI_Cloud4/glove.twitter.27B.25d.txt 

Use the following command on terminal to run sentiment_training.py

python sentiment_training.py --train "training/" --eval "eval/" --dev "dev/" --num_epoch 1 --config "training_config.json" --model_output_dir "."
