mkdir -p data_pt
cd data_pt || exit 1

#wget -O ptbxl.zip "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip"

#sudo apt-get update
#sudo apt-get install -y aria2

aria2c -c -x 16 -s 16 -k 1M -o ptbxl.zip "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip"
unzip ptbxl.zip
# rm ptbxl.zip
cd ..

