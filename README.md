I try to generate new pokemons using a relatively small dataset (455 images of existing pokemons)
using a Deep Convolutional Generative Adversarial Networks (DCGAN) with PyTorch.

Examples from dataset:
![Alt real_pokemon](images/real/real1.png?raw=false "pokemon")
![Alt real_pokemon](images/real/real2.png?raw=true "pokemon")
![Alt real_pokemon](images/real/real3.png?raw=true "pokemon")

I began from a random noise and this is the result after 10 epochs (need more fine tuning):

![Alt real_pokemon](images/fake/fake_samples_epoch_010.png?raw=true "fake")

25 epochs:

![Alt real_pokemon](images/fake/fake_samples_epoch_025.png?raw=true "fake")

46 epochs:

![Alt real_pokemon](images/fake/fake_samples_epoch_046.png?raw=true "fake")

66 epochs:

![Alt real_pokemon](images/fake/fake_samples_epoch_066.png?raw=true "fake")

99 epochs:

![Alt real_pokemon](images/fake/fake_samples_epoch_099.png?raw=true "fake")

