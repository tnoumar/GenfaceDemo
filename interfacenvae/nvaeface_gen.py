import numpy as np
import torch
import random
import string
import matplotlib.pyplot as plt
import os
CODE_DIR = "/content/GenfaceDemo/interfacenvae"
os.chdir(f"{CODE_DIR}")

import nvae

tmp_download_dir = "/content/downloaded_imgs/"
if not os.path.exists(tmp_download_dir):
  os.mkdir(tmp_download_dir)



def inference_nvae(model, download_dir):
    #generate random temperature value in 0.7 0.75
    t= 0.76# random value around 0.75 (tested and gives good resulst)
    with torch.no_grad():
        logits = model.sample(1, t)
        output = model.decoder_output(logits)
        output_img = output.sample()    
        output_img = output_img.permute(0, 2, 3, 1).squeeze().cpu().numpy()
        img_name = "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
        plt.imsave(tmp_download_dir + img_name + ".jpg", output_img)
        print("\n" + tmp_download_dir + img_name + ".jpg")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nvae.load_pretrained_model("celeba_256a", CODE_DIR+"/")
    model=model.to(device)
    inference_nvae(model, tmp_download_dir)