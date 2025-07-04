
seq_len=96
pred_len=192
model_name=SVQ

root_path_name='../../../dataset/electricity/'
data_path_name=electricity.csv
model_id_name=ECL
data_name='custom'

random_seed=2021
python -u ../../../runner.py \
        --seed $random_seed \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --model_id $model_id_name_$seq_len'_'$pred_len \
        --model $model_name \
        --data_name $model_id_name\
        --data $data_name \
        --features M \
        --seq_len $seq_len \
        --label_len 48 \
        --pred_len $pred_len \
        --enc_in 321\
        --dec_in 321\
        --c_out  321\
        --e_layers_c 2 \
        --n_heads_c 8 \
        --d_model_c 512 \
        --d_ff 512\
        --dropout 0.2\
        --fc_dropout 0.2\
        --head_dropout 0\
        --depth 2\
        --d_model_d 128\
        --num_workers 4\
        --itr 1\
        --train_epochs 100\
        --timesteps 100\
        --batch_size 4\
        --test_batch_size 4\
        --des 'Exp'\
        --lradj 'type1'\
        --denoise_model 'PatchDN'\
        --kernel_size 15\
        --fourier_factor 1.0\
        --svq 1 \
        --wFFN 0 \
        --num_codebook 2\
        --codebook_size 256 \
        --type_sample 'DPM_solver'\
        --DPMsolver_step 20\
        --gpu 0 \
        --parameterization "x_start"\
        --bias \
