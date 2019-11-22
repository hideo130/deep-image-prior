import matplotlib.pyplot as plt 
import matplotlib
import matplotlib.font_manager as fon
from glob import glob
from pathlib import Path
import pickle

def plot_config():
    del fon.weight_dict['roman']
    matplotlib.font_manager._rebuild()
    plt.rcParams['font.family'] = 'Times New Roman' #全体のフォントを設定

    #日本語なら
    # plt.rcParams['font.family']= 'sans-serif'
    # plt.rcParams['font.sans-serif'] = ['Arial']


    plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
    plt.rcParams["font.size"] = 15 # 全体のフォントサイズが変更されます。

    # 軸関連
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    # plt.rcParams['xtick.major.width'] = 1.2
    # plt.rcParams['ytick.major.width'] = 1.2
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['axes.grid']=True  # make grid
    plt.rcParams['grid.linestyle']='--'
    plt.rcParams['grid.linewidth'] = 0.3

    #凡例関連

def my_plot(losses,xlabel):
        
    fig = plt.figure()
    fig_1 = fig.add_subplot(111)
    
    for loss in losses:
        loss_name = Path(loss).stem
        if 'L1' in loss_name:
            with loss.read_bytes() as f:
                L1 = pickle.load(f)
        elif 'GAN' in loss_name:
            with loss.read_bytes() as f:
                gan = pickle.load(f)
        else:
            with loss.read_bytes() as f:
                loss = pickle.load(f)    
            fig_1.plot(loss,label=loss_name)

    fig_1.set_xlabel(xlabel)
    fig_1.set_ylabel(r"$loss$")


    fig_1.legend()

    save_name = target_dir.joinpath("%s_D.png"%(xlabel) )
    fig.savefig(str(save_name), bbox_inches="tight", pad_inches=0.05,dpi=300)

    fig_1.plot(gan,label='G_GAN_epoch')
    fig_1.legend()
    save_name = target_dir.joinpath("%s_without_L1.png"%(xlabel) )
    fig.savefig(save_name, bbox_inches="tight", pad_inches=0.05,dpi=300)

    fig_1.plot(L1,label='G_L1_it')
    fig_1.legend()
    fig.savefig( '../HEtoMT_various_scale/%d/Result/ite.png'%(resol), bbox_inches="tight", pad_inches=0.05,dpi=300)




def plot_loss(target_dir,loss_name,gan_mode):
    target_dir = Path(target_dir)
    plot_config()

    second_plot = ["G_A","G_B","G_GAN"]
    third_plot = ["G_L1"] if gan_mode == "pix2pix" else ["idt","cycle"]
    #iteratiすべてのlossをplot
    it_losses = target_dir.glob('*_it.pkl')

    for loss in plot_loss:
        if (loss not in second_plot ) and  (loss not in third_plot ):
            pass
            


def plot_cycle_loss(target_dir):
    pass
        




def plot_pix2pix_loss(target_dir):
    target_dir = Path(target_dir)
    # print(target_dir)
    it_losses = target_dir.glob('*_it.pkl')

    #configからloss dirを指定して，実行できるようにしたい．
    plot_config()

    
    fig = plt.figure()
    fig_1 = fig.add_subplot(111)

    for loss in it_losses:
        loss = str(loss)
        loss_name = Path(loss).stem
        if 'L1' in loss_name:
            with open(loss,"rb") as f:
                L1 = pickle.load(f)

        elif 'GAN' in loss_name:
            with open(loss,"rb") as f:
                gan = pickle.load(f)
        else:
            print("write %s"%(loss_name))
            with open(loss,"rb") as f:
                loss = pickle.load(f)    
            fig_1.plot(loss,label=loss_name)

    
    fig_1.set_xlabel(r"$itration$")
    fig_1.set_ylabel(r"$loss$")

    fig_1.legend()
    save_name = target_dir.joinpath("D_it.png")
    fig.savefig( str(save_name), bbox_inches="tight", pad_inches=0.05,dpi=300)

    fig_1.plot(gan,label='G_GAN_it')
    fig_1.legend()
    
    save_name = target_dir.joinpath("ite_without_L1.png")
    fig.savefig( str(save_name), bbox_inches="tight", pad_inches=0.05,dpi=300)


    fig_1.plot(L1,label='G_L1_it')
    fig_1.legend()
    save_name = target_dir.joinpath("ite.png")
    fig.savefig( str(save_name), bbox_inches="tight", pad_inches=0.05,dpi=300)




    epoch_losses = target_dir.glob('*epoch_last.pkl')
    fig = plt.figure()
    fig_1 = fig.add_subplot(111)

    for loss in epoch_losses:
        loss = str(loss)
        loss_name = Path(loss).stem
        if 'L1' in loss_name:
            with open(loss,"rb") as f:
                L1 = pickle.load(f)

        elif 'GAN' in loss_name:
            with open(loss,"rb") as f:
                gan = pickle.load(f)
        else:
            print("write %s"%(loss_name))
            with open(loss,"rb") as f:
                loss = pickle.load(f)    
            fig_1.plot(loss,label=loss_name)

    
    
    fig_1.set_xlabel(r"$epoch$")
    fig_1.set_ylabel(r"$loss$")

    
    fig_1.legend()
    save_name = target_dir.joinpath("D_epoch.png")
    fig.savefig( str(save_name), bbox_inches="tight", pad_inches=0.05,dpi=300)

    fig_1.plot(gan,label='G_GAN_epoch')
    fig_1.legend()
    
    save_name = target_dir.joinpath("epoch_without_L1.png")
    fig.savefig( str(save_name), bbox_inches="tight", pad_inches=0.05,dpi=300)


    fig_1.plot(L1,label='G_L1_epoch')
    fig_1.legend()
    save_name = target_dir.joinpath("epoch.png")
    fig.savefig( str(save_name), bbox_inches="tight", pad_inches=0.05,dpi=300)

if __name__ == "__main__":
    target_dir = "../checkpoints/dec_unet/MT/512/Result"
    plot_pix2pix_loss(target_dir)