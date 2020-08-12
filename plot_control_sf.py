import numpy as np
import pickle
import matplotlib.pyplot as plt

## k4
#
with open('./b_1_2_sf_k4_centralized_controlrate_5_stubborn.pk', 'rb') as f:
    stubborn = pickle.load(f)

with open('./b_1_2_sf_k4_centralized_controlrate_5_random_same.pk', 'rb') as f:
    random_same = pickle.load(f)

with open('./b_1_2_sf_k4_centralized_controlrate_5_wsls.pk', 'rb') as f:
    wsls = pickle.load(f)

with open('./b_1_2_sf_k4_centralized_controlrate_5_tit_tat.pk', 'rb') as f:
    tit = pickle.load(f)
with open('./b_1_2_sf_coor_freq_without_control.pk', 'rb') as f:
    nocontrol = pickle.load(f)

## k6
#
# with open('./b_1_2_sf_k6_decentralized_controlrate_5_stubborn.pk', 'rb') as f:
#     stubborn = pickle.load(f)
#
# with open('./b_1_2_sf_k6_decentralized_controlrate_5_random_same.pk', 'rb') as f:
#     random_same = pickle.load(f)
#
# with open('./b_1_2_sf_k6_decentralized_controlrate_5_wsls.pk', 'rb') as f:
#     wsls = pickle.load(f)
#
# with open('./b_1_2_sf_k6_decentralized_controlrate_5_tit_tat.pk', 'rb') as f:
#     tit = pickle.load(f)
#
# with open('./b_1_2_sf_k6_coor_freq_without_control.pk', 'rb') as f:
#     nocontrol = pickle.load(f)

plt.figure()

# plt.title(r"evolution of cooperation in lattice(fraction=5%,net=10,Control=10,Rep=5,G=20000)")
plt.title(r"evolution of cooperation on scale free network(fraction=5%,k=4,centralized control)")
plt.xlabel("b")
plt.ylabel(r"frequency probability $f_c$")
plt.xlim(1, 1.9)
plt.ylim((-0.05, 1.05))

plt.plot(random_same[0], [np.mean(array) for array in random_same[1]],
         color="b", label='Random_same')
plt.plot(stubborn[0], [np.mean(array) for array in stubborn[1]],
         color="r", label='Stubborn')
plt.plot(wsls[0], [np.mean(array) for array in wsls[1]],
         color="g", label='WSLS')
plt.plot(tit[0], [np.mean(array) for array in tit[1]],
         color="gold", label='TFT')
plt.plot(nocontrol[0], [np.mean(array) for array in nocontrol[1]], "--",
         color="grey", label='WithoutControl')

# plt.plot(random_same[0], [np.mean(array, axis=(1, 2)) for array in random_same[1]], 'o',
#          color="b")
# plt.plot(stubborn[0], [np.mean(array, axis=(1, 2)) for array in stubborn[1]], 'o',
#          color="r")
# plt.plot(wsls[0], [np.mean(array, axis=(1, 2)) for array in wsls[1]], 'o',
#          color="g")
# plt.plot(tit[0], [np.mean(array, axis=(1, 2)) for array in tit[1]], 'o',
#          color="gold")
# plt.plot(nocontrol[0], [np.mean(array, axis=1) for array in nocontrol[1]], "o",
#          color="grey")

plt.legend(loc="best")
print(nocontrol[1])

b_index = (1.9 - 1) / 0.1

plt.show()
