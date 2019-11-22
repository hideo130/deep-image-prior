
import subprocess

# CUDA_VISIBLE_DEVICES=2 python3 ../../test.py --dataroot ../../datasets/L159/25/CK19andHE --checkpoints_dir ../../checkpoints/L159_HEtoCK19_cycle --name Result --model cycle_gan --netG unet_256 --norm instance --gan_mode lsgan --num_threads 2 --num_D 3 --gpu_ids 0  --dataset_mode aligned
for i in range(10,205,5):  
# for i in range(5,10,5):  
    shell = "CUDA_VISIBLE_DEVICES=0 python3 ../test.py --dataroot ../datasets/MTandHE/512 --checkpoints_dir ../checkpoints/dec_unet/MT/512 --results_dir ../checkpoints/dec_unet/MT/512 --name Result --model dec_unet --norm instance    --num_threads 2  --gpu_ids 0 --epoch %d"%(i)
    # shell = shell.split(' ')
    print(shell)
    subprocess.run(shell,shell=True)
    # print(proc.stdout.decode("utf8"))
