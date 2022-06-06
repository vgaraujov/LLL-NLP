cd LLL-NLP/EM-in-LLL-QA

export DATA_DIR=/home/vgaraujo/storage/Datasets
export ORDER=1

if [ "$ORDER" == "1" ]; then
TASKS="${DATA_DIR}/QuAC ${DATA_DIR}/TriviaQA/Web ${DATA_DIR}/TriviaQA/Wiki ${DATA_DIR}/SQuAD1.1"
elif [ "$ORDER" == "2" ]; then
TASKS="${DATA_DIR}/SQuAD1.1 ${DATA_DIR}/TriviaQA/Wiki ${DATA_DIR}/QuAC ${DATA_DIR}/TriviaQA/Web"
elif [ "$ORDER" == "3" ]; then
TASKS="${DATA_DIR}/TriviaQA/Web ${DATA_DIR}/TriviaQA/Wiki ${DATA_DIR}/SQuAD1.1 ${DATA_DIR}/QuAC"
elif [ "$ORDER" == "4" ]; then
TASKS="${DATA_DIR}/TriviaQA/Wiki ${DATA_DIR}/QuAC ${DATA_DIR}/TriviaQA/Web ${DATA_DIR}/SQuAD1.1"
fi

NAME="output_${ORDER}"
python train.py --tasks $TASKS --output_dir $NAME
python test.py --output_dir $NAME