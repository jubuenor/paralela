import numpy as np
import matplotlib.pyplot as mp

num = [1,2,4,8,16]

# N = 128

# Video: japan.mp4
# Duracion: 00:01:00

#times = [[55.259958, 54.829866, 54.843469],
        #[0.004797, 0.004588, 0.015015],
        #[0.003112, 0.014492, 0.002976], 
        #[0.005580, 0.002812, 0.002606],
        #[0.002789, 0.002695, 0.003789]
        #]

# openmp

times = [[55.259958, 54.829866, 54.843469], # 80.671145
        [43.178546, 43.309115, 43.484728], 
        [29.809299, 30.944692, 30.741333],
        [27.600829, 28.029610, 27.824460],
        [26.496024, 26.629711, 27.446805]
       ]

time =np.array([sum(t)/len(t)for t in times])

mp.plot(num,time,"ro")
mp.plot(num,time,"r-")
mp.title("Tiempo de ejecucion")
mp.xlabel("Numero de hilos")
mp.ylabel("Tiempo (s)")
mp.grid(True)
mp.show()