cd LLL-NLP/EM-in-LLL-TC

export DATA_DIR=/home/vgaraujo/storage/Datasets/TextClassificationDatasets
export ORDER=1

if [ "$ORDER" == "1" ]; then
TASKS="${DATA_DIR}/yelp_review_full_csv ${DATA_DIR}/ag_news_csv ${DATA_DIR}/dbpedia_csv ${DATA_DIR}/amazon_review_full_csv ${DATA_DIR}/yahoo_answers_csv"
elif [ "$ORDER" == "2" ]; then
TASKS="${DATA_DIR}/dbpedia_csv ${DATA_DIR}/yahoo_answers_csv ${DATA_DIR}/ag_news_csv ${DATA_DIR}/amazon_review_full_csv ${DATA_DIR}/yelp_review_full_csv"
elif [ "$ORDER" == "3" ]; then
TASKS="${DATA_DIR}/yelp_review_full_csv ${DATA_DIR}/yahoo_answers_csv ${DATA_DIR}/amazon_review_full_csv ${DATA_DIR}/dbpedia_csv ${DATA_DIR}/ag_news_csv"
elif [ "$ORDER" == "4" ]; then
TASKS="${DATA_DIR}/ag_news_csv ${DATA_DIR}/yelp_review_full_csv ${DATA_DIR}/amazon_review_full_csv ${DATA_DIR}/yahoo_answers_csv ${DATA_DIR}/dbpedia_csv"
fi

NAME="output_${ORDER}"
python train.py --tasks $TASKS --output_dir $NAME
python test.py --output_dir $NAME