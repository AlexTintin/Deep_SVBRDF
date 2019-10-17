
import torchvision

def matplotlib_imshow(img, plt, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def f(x):
    return x

def preprocessing(img_name, trans_all):
  # get some random training images
  image = io.imread(img_name)
  input = image[:,:288,:]
  normals = image[:,288:2*288,:]
  diffuse = image[:,2*288:3 * 288,:]
  roughness = image[:,3*288:4 * 288,:]
  specular = image[:,4*288:5 * 288,:]
  #label = np.concatenate((normals,diffuse,roughness,specular),axis = 2)
  if trans_all:
      input_t = trans_all(input)
      normals_t = trans_all(normals)
      sample = {'input': input_t, 'label': normals_t}
  else:
      sample = {'input': input, 'label': normals}
  images, labels = sample["input"].unsqueeze(0), sample["label"].unsqueeze(0)
  return images, labels

def visualisation_finale(images, labels, sortie_to_plot, plt):
  # get some random training images
  # create grid of images
  img_grid = torchvision.utils.make_grid(images)
  img_grid_labels = torchvision.utils.make_grid(labels)
  img_grid_sortie_to_plot = torchvision.utils.make_grid(sortie_to_plot)
  # show images
  plt.figure(figsize=(25, 25))  # create a plot figure
  #create the first of two panels and set current axis
  plt.subplot(1, 3, 1)
  matplotlib_imshow(img_grid, plt, one_channel=False)
  plt.subplot(1, 3, 2)
  matplotlib_imshow(img_grid_labels, plt, one_channel=False)
  plt.subplot(1, 3, 3)
  matplotlib_imshow(img_grid_sortie_to_plot.cpu().detach(), plt, one_channel=False)
  plt.show()


def pipeline_all(img_name, trans_all, the_model, plt):
  images, labels = preprocessing(img_name, trans_all)
  x_latent = the_model.encoder(images.float().to(device))
  sortie_to_plot = the_model.decoder(x_latent.float().to(device))
  visualisation_finale(images, labels, sortie_to_plot, plt)

def pipeline_inter(dico, images, labels, the_model, plt):
  x_latent_liste = []
  for index in range(len(dico)):
    x_latent_liste.append(dico[index].result)
  x_latent = trans_all(np.array([x_latent_liste]))
  sortie_to_plot = the_model.decoder(x_latent.float().to(device))
  visualisation_finale(images, labels, sortie_to_plot, plt)
