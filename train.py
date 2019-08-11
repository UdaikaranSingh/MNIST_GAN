import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
import random
import torchvision.transforms as transforms

#######################################
# Importing and setting up networks 
#######################################

from normal_model import Generator, Discriminator

cuda_available = torch.cuda.is_available()
latent_size = 50
batch_size = 16

if cuda_available:
    print("Using GPU")
else:
    print("Using CPU")

if cuda_available:
	generator = Generator(batch_size, latent_size).cuda()
	discriminator = Discriminator(batch_size, latent_size).cuda()

	if os.path.exists("generator_model_normal.pth"):
		print("Loading in Models")
		generator.load_state_dict(torch.load("generator_model_normal.pth"))
		discriminator.load_state_dict(torch.load("discriminator_model_normal.pth"))
else:
	generator = Generator(batch_size, latent_size)
	discriminator = Discriminator(batch_size, latent_size)

	if os.path.exists("generator_model_normal.pth"):
		print("Loading in Models")
		generator.load_state_dict(torch.load("generator_model_normal.pth"))
		discriminator.load_state_dict(torch.load("discriminator_model_normal.pth"))

##################################################
# Definining Hyperparameters of Training Procedure
##################################################
random.seed(1)
learning_rate = 0.2e-4
beta1 = 0.5
beta2 = 0.999
num_epochs = 100
epsilon = 1e-8

optimizer_G = torch.optim.Adam(generator.parameters(), lr= learning_rate, betas= (beta1, beta2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr= learning_rate, betas= (beta1, beta2))

transform = transforms.Compose([transforms.Resize((28, 28)),
	transforms.ToTensor(), 
	transforms.Normalize([0.5], [0.5])
])

os.makedirs("../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data/mnist",
        train=True,
        download=True,
        transform=transform,
    ),
    batch_size=batch_size,
    shuffle=True,
)

adversarial_loss = torch.nn.BCELoss()
##################################################
# Training Procedure
##################################################

Tensor = torch.cuda.FloatTensor if cuda_available else torch.FloatTensor

loss_tracker = []

for epoch in range(num_epochs):

	count = 0

	print("epoch: " + str(epoch + 1))

	for i, (imgs, labels) in enumerate(dataloader):

		valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
		fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

		########################
        # Generator Step
        ########################

		real_imgs = Variable(imgs.type(Tensor))

		generator.zero_grad()

		if cuda_available:
			z = Variable(torch.rand(batch_size, latent_size).type(torch.cuda.FloatTensor))
		else:
			z = Variable(torch.rand(batch_size, latent_size).type(torch.FloatTensor))

		gen_imgs = generator(z)

		g_loss = adversarial_loss(discriminator(gen_imgs), valid)

		g_loss.backward()
		optimizer_G.step()

		########################
		# Discriminator Step
		########################

		discriminator.zero_grad()

		real_loss = adversarial_loss(discriminator(real_imgs), valid)
		fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
		d_loss = (real_loss + fake_loss) / 2

		d_loss.backward()
		optimizer_D.step()

	path = "generator_model_normal.pth"
	torch.save(generator.state_dict(), path)
	path = "discriminator_model_normal.pth"
	torch.save(discriminator.state_dict(), path)





