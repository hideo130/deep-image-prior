export CUDA_VISIBLE_DEVICES=0,1
python3 ../train_unit.py --dataroot ../../cycle-gan/datasets/L159/original/CK19andHE/observe_dataset/  --checkpoints_dir ../checkpoints/unit/ --name debug2 --dataset_mode aligned --gpu_ids 0,1 --batch_size 4 --num_D 1 --num_threads 4
