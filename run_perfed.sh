#!/usr/bin/env bash
python main.py --algo='me' --dataset='synthetic_alpha0_beta0_niid_balance' \
		--model='logistic' --criterion='celoss' --wd=0.001 \
		--device=0 --num_round=5 --eval_every=5 \
		--clients_per_round=10 --batch_size=64 \
		--num_epoch=50 --num_local_round=1 \
		--local_lr=0.01 --person_lr=0.005 \
		--lamda=5 --q=2 --d=21 \
		--seed=1
		
python main.py --algo='me' --dataset='synthetic_alpha0_beta0_niid_balance' \
		--model='logistic' --criterion='celoss' --wd=0.001 \
		--device=0 --num_round=5 --eval_every=5 \
		--clients_per_round=10 --batch_size=64 \
		--num_epoch=50 --num_local_round=1 \
		--local_lr=0.01 --person_lr=0.005 \
		--lamda=1 --q=2 --d=21 \
		--seed=1 --server='robust_server' --mali_frac=0.2 --attack='sign_flip'
		
python main.py --algo='lp' --dataset='synthetic_alpha0_beta0_niid_balance' \
		--model='logistic' --criterion='celoss' --wd=0.001 \
		--device=0 --num_round=20 --eval_every=5 \
		--clients_per_round=10 --batch_size=64 \
		--num_epoch=50 --num_local_round=1 \
		--local_lr=0.01 --person_lr=0.005 \
		--lamda=1 --q=2 --d=21 \
		--seed=1 --p=1\
		--server='robust_server' --mali_frac=0.2 --attack='gaussian' --aggr='mean'
		
python main.py --algo='ditto' --dataset='synthetic_alpha0_beta0_niid_balance' \
		--model='logistic' --criterion='celoss' --wd=0.001 \
		--device=0 --num_round=5 --eval_every=5 \
		--clients_per_round=10 --batch_size=64 \
		--num_epoch=50 --num_local_round=1 \
		--local_lr=0.01 --person_lr=0.005 \
		--lamda=1 --q=3 --d=21 \
		--seed=1 --p=1\
		--server='robust_server' --mali_frac=0.2 --attack='gaussian' --aggr='krum'

		
python main.py --algo='me' --dataset='synthetic_alpha0_beta0_niid_balance' \
		--model='logistic' --criterion='celoss' --wd=0.001 \
		--device=0 --num_round=20 --eval_every=5 \
		--clients_per_round=10 --batch_size=64 \
		--num_epoch=50 --num_local_round=1 \
		--local_lr=0.01 --person_lr=0.005 \
		--lamda=15 --q=2 --d=21 \
		--seed=1
		
python main.py --algo='me' --dataset='synthetic_alpha0_beta0_niid_balance' \
		--model='2nn' --criterion='celoss' --wd=0.001 \
		--device=0 --num_round=20 --eval_every=5 \
		--clients_per_round=10 --batch_size=64 \
		--num_epoch=50 --num_local_round=1 \
		--local_lr=0.01 --person_lr=0.005 \
		--lamda=1 --q=2 --d=21 \
		--seed=1
		
python main.py --algo='me_fair' --dataset='synthetic_alpha0_beta0_niid_balance' \
		--model='logistic' --criterion='celoss' --wd=0.001 \
		--device=0 --num_round=20 --eval_every=5 \
		--clients_per_round=10 --batch_size=64 \
		--num_epoch=50 --num_local_round=1 \
		--local_lr=0.05 --person_lr=0.01 \
		--lamda=5 --q=2 --d=21 \
		--seed=1
		
python main.py --algo='me_fair' --dataset='synthetic_alpha0_beta0_niid_balance' \
		--model='logistic' --criterion='celoss' --wd=0.001 \
		--device=0 --num_round=20 --eval_every=5 \
		--clients_per_round=10 --batch_size=64 \
		--num_epoch=50 --num_local_round=1 \
		--local_lr=0.05 --person_lr=0.01 \
		--lamda=5 --q=5 --d=21 \
		--seed=1
		
python main.py --algo='me_fair' --dataset='synthetic_alpha0_beta0_niid_balance' \
		--model='logistic' --criterion='celoss' --wd=0.001 \
		--device=0 --num_round=20 --eval_every=5 \
		--clients_per_round=10 --batch_size=64 \
		--num_epoch=50 --num_local_round=1 \
		--local_lr=0.05 --person_lr=0.01 \
		--lamda=15 --q=2 --d=21 \
		--seed=1
		
python main.py --algo='me_fair' --dataset='synthetic_alpha0_beta0_niid_balance' \
		--model='logistic' --criterion='celoss' --wd=0.001 \
		--device=0 --num_round=20 --eval_every=5 \
		--clients_per_round=10 --batch_size=64 \
		--num_epoch=50 --num_local_round=1 \
		--local_lr=0.05 --person_lr=0.01 \
		--lamda=15 --q=5 --d=21 \
		--seed=1
		
python main.py --algo='me_fair' --dataset='synthetic_alpha0_beta0_niid_balance' \
		--model='logistic' --criterion='celoss' --wd=0.001 \
		--device=0 --num_round=20 --eval_every=5 \
		--clients_per_round=10 --batch_size=64 \
		--num_epoch=50 --num_local_round=1 \
		--local_lr=0.05 --person_lr=0.01 \
		--lamda=1 --q=2 --d=21 \
		--seed=1
		
python main.py --algo='me_fair' --dataset='synthetic_alpha0_beta0_niid_balance' \
		--model='logistic' --criterion='celoss' --wd=0.001 \
		--device=0 --num_round=20 --eval_every=5 \
		--clients_per_round=10 --batch_size=64 \
		--num_epoch=50 --num_local_round=1 \
		--local_lr=0.05 --person_lr=0.01 \
		--lamda=1 --q=5 --d=21 \
		--seed=1
		
python main.py --algo='proj' --dataset='synthetic_alpha0_beta0_niid_balance' \
		--model='logistic' --criterion='celoss' --wd=0.001 \
		--device=0 --num_round=20 --eval_every=5 \
		--clients_per_round=10 --batch_size=64 \
		--num_epoch=50 --num_local_round=1 \
		--local_lr=0.05 --person_lr=0.01 \
		--lamda=5 --q=2 --d=21 \
		--seed=1

python main.py --algo='proj' --dataset='synthetic_alpha0_beta0_niid_balance' \
		--model='logistic' --criterion='celoss' --wd=0.001 \
		--device=0 --num_round=20 --eval_every=5 \
		--clients_per_round=10 --batch_size=64 \
		--num_epoch=50 --num_local_round=1 \
		--local_lr=0.05 --person_lr=0.01 \
		--lamda=15 --q=2 --d=21 \
		--seed=1
		
python main.py --algo='proj' --dataset='synthetic_alpha0_beta0_niid_balance' \
		--model='logistic' --criterion='celoss' --wd=0.001 \
		--device=0 --num_round=20 --eval_every=5 \
		--clients_per_round=10 --batch_size=64 \
		--num_epoch=50 --num_local_round=1 \
		--local_lr=0.05 --person_lr=0.01 \
		--lamda=1 --q=2 --d=21 \
		--seed=1
		
python main.py --algo='ditto' --dataset='synthetic_alpha0_beta0_niid_balance' \
		--model='logistic' --criterion='celoss' --wd=0.001 \
		--device=0 --num_round=20 --eval_every=5 \
		--clients_per_round=10 --batch_size=64 \
		--num_epoch=50 --num_local_round=1 \
		--local_lr=0.05 --person_lr=0.01 \
		--lamda=5 --q=2 --d=21 \
		--seed=1
		
python main.py --algo='ditto' --dataset='synthetic_alpha0_beta0_niid_balance' \
		--model='logistic' --criterion='celoss' --wd=0.001 \
		--device=0 --num_round=20 --eval_every=5 \
		--clients_per_round=10 --batch_size=64 \
		--num_epoch=50 --num_local_round=1 \
		--local_lr=0.05 --person_lr=0.01 \
		--lamda=15 --q=2 --d=21 \
		--seed=1
		
python main.py --algo='ditto' --dataset='synthetic_alpha0_beta0_niid_balance' \
		--model='logistic' --criterion='celoss' --wd=0.001 \
		--device=0 --num_round=20 --eval_every=5 \
		--clients_per_round=10 --batch_size=64 \
		--num_epoch=50 --num_local_round=1 \
		--local_lr=0.05 --person_lr=0.01 \
		--lamda=1 --q=2 --d=21 \
		--seed=1
		
python main.py --algo='proj_fair' --dataset='synthetic_alpha0_beta0_niid_balance' \
		--model='logistic' --criterion='celoss' --wd=0.001 \
		--device=0 --num_round=20 --eval_every=5 \
		--clients_per_round=10 --batch_size=64 \
		--num_epoch=50 --num_local_round=1 \
		--local_lr=0.05 --person_lr=0.01 \
		--lamda=10 --q=3 --d=21 \
		--seed=1
		
python main.py --algo='me' --dataset='mnist_all_data_0_equal_niid' \
		--model='2nn' --criterion='celoss' --wd=0.001 \
		--device=0 --num_round=5 --eval_every=5 \
		--clients_per_round=10 --batch_size=64 \
		--num_epoch=10 --num_local_round=1 \
		--local_lr=0.01 --person_lr=0.005 \
		--lamda=0.2 --q=2 --d=600 \
		--seed=1
		
python main.py --algo='ditto' --dataset='mnist_all_data_0_equal_niid' \
		--model='2nn' --criterion='celoss' --wd=0.001 \
		--device=0 --num_round=20 --eval_every=5 \
		--clients_per_round=10 --batch_size=64 \
		--num_epoch=10 --num_local_round=1 \
		--local_lr=0.01 --person_lr=0.005 \
		--lamda=0.2 --q=2 --d=600 \
		--seed=1
		
python main.py --algo='me_fair' --dataset='mnist_all_data_0_equal_niid' \
		--model='2nn' --criterion='celoss' --wd=0.001 \
		--device=0 --num_round=20 --eval_every=5 \
		--clients_per_round=10 --batch_size=64 \
		--num_epoch=10 --num_local_round=1 \
		--local_lr=0.01 --person_lr=0.005 \
		--lamda=0.1 --q=1 --d=600 \
		--seed=1
		
python main.py --algo='proj' --dataset='mnist_all_data_0_equal_niid' \
		--model='2nn' --criterion='celoss' --wd=0.001 \
		--device=0 --num_round=20 --eval_every=5 \
		--clients_per_round=10 --batch_size=64 \
		--num_epoch=20 --num_local_round=1 \
		--local_lr=0.01 --person_lr=0.005 \
		--lamda=0.1 --q=2 --d=100 \
		--seed=1

python main.py --algo='proj_fair' --dataset='mnist_all_data_0_equal_niid' \
		--model='2nn' --criterion='celoss' --wd=0.001 \
		--device=0 --num_round=100 --eval_every=5 \
		--clients_per_round=10 --batch_size=64 \
		--num_epoch=10 --num_local_round=5 \
		--local_lr=0.01 --person_lr=0.005 \
		--lamda=10 --q=2 --d=600 \
		--seed=1

python main.py --algo='sketch' --dataset='synthetic_alpha0_beta0_niid_balance' \
		--model='logistic' --criterion='celoss' --wd=0.001 \
		--device=0 --num_round=20 --eval_every=5 \
		--clients_per_round=10 --batch_size=64 \
		--lamda=0.1 --seed=1 --server='server_sketch'\
		--sketchparamslargerthan=0 --p2=4 \
		--c=70 --r=20 --k=100 \
		--momentum=0.9
		
python main.py --algo='sketch' --dataset='mnist_all_data_0_equal_niid' \
		--model='2nn' --criterion='celoss' --wd=0.001 \
		--device=0 --num_round=5 --eval_every=5 \
		--clients_per_round=100 --batch_size=64 \
		--num_epoch=10 --num_local_round=1 \
		--local_lr=0.5 --seed=1 --server='server_sketch' \
		--sketchparamslargerthan=0 --p2=10 \
		--c=200 --r=80 --k=10 \
		--momentum=0.9
		
python main.py --algo='me' --dataset='mnist_all_data_0_equal_niid' \
		--model='logistic' --criterion='celoss' --wd=0.001 \
		--device=0 --num_round=20 --eval_every=5 \
		--clients_per_round=10 --batch_size=64 \
		--num_epoch=10 --num_local_round=1 \
		--local_lr=0.01 --person_lr=0.005 \
		--lamda=0.2 --q=2 --d=600 \
		--seed=1
		
python main.py --algo='lg' --dataset='synthetic_alpha0_beta0_niid_balance' \
		--model='2nn' --criterion='celoss' --wd=0.001 \
		--device=0 --num_round=5 --eval_every=5 \
		--clients_per_round=100 --batch_size=64 \
		--num_epoch=1 --num_local_round=1 \
		--local_lr=0.05 --seed=1 --server='server_lg' --num_layers_keep=1
		
python main.py --algo='me' --dataset='mnist_all_data_0_equal_niid' \
		--model='2nn' --criterion='celoss' --wd=0.001 \
		--device=0 --num_round=20 --eval_every=5 \
		--clients_per_round=10 --batch_size=64 \
		--num_epoch=20 --num_local_round=1 \
		--local_lr=0.01 --person_lr=0.005 \
		--lamda=100 --q=2 --d=100 \
		--seed=1
		
python main.py --algo='me' --dataset='synthetic_alpha0_beta0_niid_balance' \
		--model='1nn' --criterion='celoss' --wd=0.001 \
		--device=0 --num_round=5 --eval_every=5 \
		--clients_per_round=10 --batch_size=64 \
		--num_epoch=20 --num_local_round=1 \
		--local_lr=0.1 --person_lr=0.005 \
		--lamda=0.1 --q=2 --d=100 \
		--seed=1
		
python main.py --gpu --dataset 'synthetic_alpha0_beta0_niid' --clients_per_round=10 \
	--num_round=200 --num_epoch=1 --batch_size=64 --lr=0.1 \
	--device=0 --seed=1 --model='2nn' --algo='fedavg'
	
python main.py --algo='perfedavg' --dataset='synthetic_alpha0_beta0_niid_balance' \
		--model='1nn' --criterion='celoss' --wd=0.001 \
		--device=0 --num_round=5 --eval_every=5 \
		--clients_per_round=10 --batch_size=64 \
		--num_epoch=1 --num_local_round=5 \
		--local_lr=0.1 --person_lr=0.05 \
		--lamda=0.1 --q=2 --d=100 \
		--seed=1 --server='server_perfedavg'

python main.py --algo='perfedavg2' --dataset='synthetic_alpha0_beta0_niid_balance' \
		--model='1nn' --criterion='celoss' --wd=0.001 \
		--device=0 --num_round=5 --eval_every=5 \
		--clients_per_round=10 --batch_size=64 \
		--num_epoch=1 --num_local_round=5 \
		--local_lr=0.1 --person_lr=0.05 \
		--lamda=0.1 --q=2 --d=100 \
		--seed=1
		
python main.py --algo='local' --dataset='synthetic_alpha0_beta0_niid_balance' \
		--model='logistic' --criterion='celoss' --wd=0.001 \
		--device=0 --num_round=5 --eval_every=5 \
		--batch_size=64 --local_lr=0.1 --person_lr=0.05 \
		--seed=1 --server='server_local'