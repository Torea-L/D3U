### ETTh2 Dataset ###
model='iTransformer'
checkpoints='./checkpoints/all/'
root_path='/home/CICDTSM_BSZF/Datas/ETT-small/'
data_path='ETTh2.csv'
model_id='iTransformer_ETTh2_96_192'
dataset='ETTh2'

## Data Configs ##
seq_len=96
label_len=48
pred_len=192
batch_size=16
test_batch_size=1

## model config ##
d_model=128
d_ff=128

## optim config ##
learning_rate=0.0001
lradj='type1'
delta=-0.0001

python -u cond_model_main.py \
    --is_training \
    --seed 2021 \
    --checkpoints $checkpoints \
    --model $model \
    --model_id $model_id \
    --root_path $root_path \
    --data_path $data_path \
    --data $dataset \
    --seq_len $seq_len \
    --label_len $label_len\
    --pred_len $pred_len \
    --patch_len 16 \
    --stride 8 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --e_layers 2 \
    --d_layers 1 \
    --train_epochs 10 \
    --patience 5 \
    --batch_size $batch_size \
    --test_batch_size $test_batch_size\
    --d_model $d_model \
    --d_ff $d_ff \
    --dropout 0.2 \
    --learning_rate $learning_rate \
    --lradj $lradj \
    --delta $delta \
    >>logs/iTrans_M_ETTh2_ALL.log