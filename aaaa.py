import numpy as np
import os
import random



f=open("9lwa.txt","a+")
f.truncate(0)
f.write("ééé")
f.close()
f=open("9lwa.txt","r").read()
print(f)


#os.remove("9lwa.txt")