export CUDA_VISIBLE_DEVICES=0,1,2
python3 ../train_unit.py --dataroot ../../cycle-gan/datasets/L159/original/CK19andHE/256 --checkpoints_dir ../checkpoints/unit/ --name Result1 --dataset_mode aligned --gpu_ids 0,1,2  --batch_size 5 --num_threads 5 --num_D 2 
