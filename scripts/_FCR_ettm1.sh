if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

# named argument 받기
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --mn)
      MN="$2"; shift 2 ;;
    --gpu)
      GPU="$2"; shift 2 ;;
    *)
      shift ;;
  esac
done
DEFAULT_MODEL_NAME="FCRMixer"
DEFAULT_GPU=0

seq_len=336
pred_len=$seq_len
model_name=${MN:-$DEFAULT_MODEL_NAME}
gpu=${GPU:-$DEFAULT_GPU}

root_path_name=./dataset/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1

random_seed=2024
for ch in 0 1 2 3 4 5 6
do
    python3 -u cr_run_longExp.py \
      --gpu $gpu \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --des 'Exp' \
      --train_epochs 100\
      --patience 5\
      --cnn_hidden 12 \
      --cnn_kernel 15 \
      --cen_num_layers 8\
      --pct_start 0.4\
      --itr 1 --batch_size 128 --learning_rate 0.0001\
      --channel $ch\
      2>&1 | tee logs/ChannelRecovery/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$ch.log 
done