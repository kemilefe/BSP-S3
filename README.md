# BSP-S3

Setup Instructions
 -Prerequisites
 -Python 3.8 or higher
 -Django 4.x
 -PyTorch
 -Transformers library (Hugging Face)
 -Pandas

 Install dependencies:
   pip install -r requirements.txt

 Ensure the BERT model is available locally. Update the paths in nlp_utils.py:
   model = BertForSequenceClassification.from_pretrained('<path-to-bert-model>')
   tokenizer = BertTokenizer.from_pretrained('<path-to-bert-model>')

 Apply Django migrations:
   python manage.py migrate

 Run the server:
   python manage.py runserver
