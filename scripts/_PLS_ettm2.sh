if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/ChannelPrediction" ]; then
    mkdir -p ./logs/ChannelPrediction
fi

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --mn)
      MN="$2"; shift 2 ;;
    --gpu)
      GPU="$2"; shift 2 ;;
    --kmin)
      KMIN="$2"; shift 2 ;;
    --kmax)
      KMAX="$2"; shift 2 ;;
    --klist)
      KLIST="$2"; shift 2 ;;
    *)
      shift ;;
  esac
done

DEFAULT_MODEL_NAME="PLS"
DEFAULT_GPU=0
DEFAULT_KMIN=2
DEFAULT_KMAX=16

seq_len=336
pred_len=$seq_len
model_name=${MN:-$DEFAULT_MODEL_NAME}
gpu=${GPU:-$DEFAULT_GPU}
kmin=${KMIN:-$DEFAULT_KMIN}
kmax=${KMAX:-$DEFAULT_KMAX}
klist=${KLIST:-""}

root_path_name=./dataset/
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2

random_seed=2024

for ch in 0 1 2 3 4 5 6
do
    cmd=(
      python3 -u cp_run_pls.py
      --gpu $gpu
      --random_seed $random_seed
      --root_path $root_path_name
      --data_path $data_path_name
      --model_id ${model_id_name}_${seq_len}_${pred_len}
      --model $model_name
      --data $data_name
      --features M
      --seq_len $seq_len
      --pred_len $pred_len
      --enc_in 7
      --batch_size 128
      --channel $ch
    )

    if [ -n "$klist" ]; then
      cmd+=(--pls_components_list "$klist")
    else
      cmd+=(--pls_components_min $kmin --pls_components_max $kmax)
    fi

    "${cmd[@]}" 2>&1 | tee logs/ChannelPrediction/${model_name}_${model_id_name}_${seq_len}_${pred_len}_${ch}.log
done