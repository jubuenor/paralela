import numpy as np
import matplotlib.pyplot as mp

num = [1,2,4,8,16]

#N = 128

#times = [[0.008836, 0.008811, 0.008797],
        #[0.004797, 0.004588, 0.015015],
        #[0.003112, 0.014492, 0.002976], 
        #[0.005580, 0.002812, 0.002606],
        #[0.002789, 0.002695, 0.003789]
        #]

#N = 256

#times = [[0.078417, 0.077365, 0.076943],
        #[0.040288, 0.039107, 0.039300], 
        #[0.022242, 0.020742, 0.024966],
        #[0.020490, 0.023320, 0.020969],
        #[0.028269, 0.022182, 0.023225]
        #]

#N = 512

#times = [[0.673425, 0.803969, 0.692345],
        #[0.341793, 0.357143, 0.338844], 
        #[0.177945, 0.188473, 0.170221],
        #[0.185384, 0.179465, 0.174871],
        #[0.187009, 0.177224, 0.176548]
       #]

N = 1024

times = [[9.152953, 9.271059, 9.818956],
        [4.919302, 5.021610, 5.113630], 
        [2.788187, 2.748063, 2.763088],
        [2.543967, 2.605562, 2.506071],
        [2.349115, 2.363338, 2.225431]
       ]



time =np.array([sum(t)/len(t)for t in times])

mp.plot(num,time,"ro")
mp.plot(num,time,"r-")
mp.title("Tiempo de ejecucion para N = "+str(N))
mp.xlabel("Numero de hilos")
mp.ylabel("Tiempo (s)")
mp.grid(True)
mp.show()