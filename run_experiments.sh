set -x

# NOTE: for actual applications, you may want to change:
#   train batch size to 4-8 (depending on the GPU you are using)
#   eval batch size to 16-32 (depending on the GPU you are using)
#   accumulation steps to 4-8 (depending on the GPU you are using)
#   num train epochs to 10-15

python download.py --model='bert-base-uncased'
python download.py --model='general_bert'
python download.py --model='medical_bert_from_scratch__longer'

# NER with original BERT
python main.py \
    --task='sequence_labelling' \
    --embedding='bert-base-uncased' \
    --do_lower_case \
    --do_train \
    --do_predict

# NER with our general BERT
python main.py \
    --task='sequence_labelling' \
    --embedding='general_bert' \
    --do_lower_case \
    --do_train \
    --do_predict

# Sentiment Analysis with our medical BERT
python main.py \
    --task='classification' \
    --embedding='medical_bert_from_scratch__longer' \
    --do_lower_case \
    --do_train \
    --do_predict

set +x
