set -e
cleanup () {
    echo "Cleaning up..."
    rm -r pkl/
    rm -r models/
    mkdir models/
    mkdir pkl/
}

cp multiData.py emnlp2017-bilstm-cnn-crf/
cp BIOF1Validation.py emnlp2017-bilstm-cnn-crf/util/
for run in 1 2 3 4 5
do
echo "---------RUN $run ------------"
python3 reMakeData.py --train_sz 80 --test_sz 20 --shuffle --multi_data
cd emnlp2017-bilstm-cnn-crf/
trap cleanup EXIT
python3 multiData.py
rm -r pkl/
mkdir pkl/
cd ..
done
