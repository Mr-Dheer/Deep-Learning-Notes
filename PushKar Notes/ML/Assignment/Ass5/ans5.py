import numpy as np
from matplotlib import pyplot as plt

# for sphere and best
vnp_NES = np.load("NES_best.npy")
vnp_CEM = np.load("CEM_best.npy")
vnp_CMAES = np.load("CMAES_best.npy")
vmin = min(len(vnp_CEM),len(vnp_NES),len(vnp_CMAES))
plt.plot(np.arange(vmin),vnp_NES[:vmin],label = "Best Fitness NES")
plt.plot(np.arange(vmin),vnp_CEM[:vmin],label = "Best Fitness CEM")
plt.plot(np.arange(vmin),vnp_CMAES[:vmin],label = "Best Fitness CMAES")
plt.legend()
plt.xlabel("Number of iterations")
plt.ylabel("Fitness score")
plt.show()
#plt.plot()

# for rastrigin and best
vnp_NES = np.load("NES_best_rastrigin.npy")
vnp_CEM = np.load("CEM_best_rastrigin.npy")
vnp_CMAES = np.load("CMAES_best_rastrigin.npy")
plt.plot(np.arange(vmin),vnp_NES[:vmin],label = "Best Fitness NES")
plt.plot(np.arange(vmin),vnp_CEM[:vmin],label = "Best Fitness CEM")
plt.plot(np.arange(vmin),vnp_CMAES[:vmin],label = "Best Fitness CMAES")
plt.legend()
plt.xlabel("Number of iterations")
plt.ylabel("Fitness score")
plt.show()

# for sphere and worst
vnp_NES = np.load("NES_worst.npy")
vnp_CEM = np.load("CEM_worst.npy")
vnp_CMAES = np.load("CMAES_worst.npy")
plt.plot(np.arange(vmin),vnp_NES[:vmin],label = "worst Fitness NES")
plt.plot(np.arange(vmin),vnp_CEM[:vmin],label = "worst Fitness CEM")
plt.plot(np.arange(vmin),vnp_CMAES[:vmin],label = "worst Fitness CMAES")
plt.legend()
plt.xlabel("Number of iterations")
plt.ylabel("Fitness score")
plt.show()


# for rastrigin and best
vnp_NES = np.load("NES_worst_rastrigin.npy")
vnp_CEM = np.load("CEM_worst_rastrigin.npy")
vnp_CMAES = np.load("CMAES_worst_rastrigin.npy")
plt.plot(np.arange(vmin),vnp_NES[:vmin],label = "worst Fitness NES")
plt.plot(np.arange(vmin),vnp_CEM[:vmin],label = "worst Fitness CEM")
plt.plot(np.arange(vmin),vnp_CMAES[:vmin],label = "worst Fitness CMAES")
plt.legend()
plt.xlabel("Number of iterations")
plt.ylabel("Fitness score")
plt.show()

