import torch
import torchvision
import os
import sys
from torch.cuda.amp import GradScaler, autocast
from SIMCLR_db import SimCLR
from Ntxent import NT_Xent
import Dataloader as dl
from Lars import LARS
sys.path.append('../')
import Backbones.ResNet as rn
import PlotUtils.PlotUtils as plot_utils


def load_optimizer(model, optim):
	# scheduler = None
	if optim == "Adam":
		optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
			optimizer,num_epochs,eta_min=0, last_epoch=-1)
	elif optim == "LARS":
		learning_rate = 0.3 * batch_size / 256
		optimizer = LARS(
			model.parameters(),
			lr = learning_rate,
			weight_decay = 1e-6)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
			optimizer, num_epochs, eta_min=0, last_epoch=-1)
	else:
		raise NotImplementedError
	return optimizer, scheduler


def save_model(epoch, model, optimizer, scheduler, loss_dict):
	if not os.path.exists("Logs"):
		os.mkdir("Logs")

	out = os.path.join("Logs","checkpoint_{}.tar".format(epoch))
	torch.save({
		'epoch': epoch,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'scheduler_state_dict': scheduler.state_dict(),
		'loss_dict': loss_dict
		}, out)

def load_model(model, optimizer, scheduler, model_path, model_only=False):	
	try:
		checkpoint = torch.load(model_path)
	except:
		print("No model file exists")
	if model_only:
		return model

	epoch = checkpoint['epoch']
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
	loss_dict = checkpoint["loss_dict"]
	return epoch, model, optimizer, scheduler, loss_dict

def plot_train_dict(train_dict, model_name):
	if not os.path.exists("Plots"):
		os.mkdir("Plots")

	save_fp_epochs = os.path.join("Plots","{}_EpochLosses".format(model_name))
	save_fp_iterations = os.path.join("Plots","{}_IterationLosses".format(model_name))
	#save_fp_epochs =  + "/" + model_name + "_EpochLosses"
	#save_fp_iterations = save_fp + "/" + model_name + "_Iterations"

	epoch_losses = train_dict["epoch_losses"]
	plot_utils.plot(x_vals=range(1, len(epoch_losses) + 1), y_vals=epoch_losses,
                    title=model_name + "\n Epochs vs. Loss", x_label="Epochs", y_label="Loss",
                    save_fp=save_fp_epochs)
	iteration_losses = train_dict["iteration_losses"]
	plot_utils.plot(x_vals=range(1, len(iteration_losses) + 1), y_vals=iteration_losses,
                     title=model_name + "\n Iterations vs. Loss", x_label="Iterations", y_label="Loss",
                     save_fp=save_fp_iterations)


def train(model, optimizer, scheduler,train_loader, criterion, num_epochs, model_path):
	
	if model_path is not None:
		epoch, model, optimizer, scheduler, loss_dict = load_model(
			model, optimizer, scheduler, model_path)
	else:
		epoch = None
		loss_dict = {"epoch_losses": [], "iteration_losses": []}
	if epoch is None:
		start_epoch = 1
	else:
		start_epoch = epoch + 1

	scaler = GradScaler()

	for epoch_num in range(start_epoch, num_epochs + 1):
		loss_epoch = 0
		loss_iteration = []
		for step, data in enumerate(train_loader):
			#print(step, ((x_i,x_j),_))
			optimizer.zero_grad()

			with autocast():
				ims = data.cuda(non_blocking=True)
				N_ = ims.shape[0]
				N = int(N_ / 2)
				x_i = ims[0:N]
				x_j = ims[N:]

				h_i,h_j,z_i,z_j = model(x_i,x_j)

				loss = criterion(z_i, z_j)

			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
			#loss.backward()
			#optimizer.step()
		
			if step % 100 ==0:
				print(f"Step[{step}/{len(list(train_loader))}]\t Loss:{loss.item()}")
			
			loss_epoch += loss.item()
			loss_iteration.append(loss.item())

		scheduler.step()
		loss_dict["epoch_losses"].extend([loss_epoch])
		loss_dict["iteration_losses"].extend(loss_iteration)

		print("\nFinished training epoch " + str(epoch_num) + " with average loss: " + str(round(loss_epoch/step, 3)))

		save_model(epoch_num, model, optimizer, scheduler, loss_dict)

	return{"epoch_losses": loss_dict["epoch_losses"], "iteration_losses":loss_dict["iteration_losses"]}

if __name__ == "__main__":
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	batch_size = 32
	temperature = 0.5
	num_epochs = 20
	model_name = "SimCLR_dual_depth"

	train_loader = dl.get_unlabeled_pair_dl(batch_size=batch_size, num_workers=4, depth_only=True)

	encoder1 = rn.get_grayscale_rn50_backbone(pre_trained=True, with_pooling=True)
	encoder2 = encoder1

	encoder1.output_dim = 2048
	n_features = encoder1.output_dim
	projection_dim = 128

	model = SimCLR(encoder1, encoder2, projection_dim, n_features)
	model.to(device)

	optimizer, scheduler = load_optimizer(model, "LARS")

	criterion = NT_Xent(batch_size, temperature)

	train_dict = train(model,optimizer,scheduler,train_loader,criterion, num_epochs, model_path=None)

	plot_train_dict(train_dict, model_name)

	
	"""
	# load pretrained model
	model_fp = os.path.join("Logs","checkpoint_25.tar")
	checkpoint = torch.load(model_fp, map_location=device)
	model.load_state_dict(checkpoint['model_state_dict'])
	model.to(device)
	"""

	