import os
import matplotlib.pyplot as plt

def plot_reliability(mean_pred,frac_pos,save_path:str,title:str):

    os.makedirs(os.path.dirname(save_path),exist_ok=True)

    plt.figure()
    plt.plot(mean_pred,frac_pos,marker="o")
    plt.plot([0,1],[0,1],linestyle="--")
    plt.xlabel("Mean predicted probablity")
    plt.ylabel("Fraction of positives")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()