set -e

cp multiTask.py emnlp2017-bilstm-cnn-crf/
cp BIOF1Validation.py emnlp2017-bilstm-cnn-crf/util/
for run in 1 2 3 4 5
do
echo "---------RUN $run ------------"
python3 reMakeData.py --train_sz 80 --test_sz 20
cd emnlp2017-bilstm-cnn-crf/
python3 multiTask.py
cd ..
done
