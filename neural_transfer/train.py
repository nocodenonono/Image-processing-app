"""
training code for style transfer
"""
from torch.optim import Adam
from vgg import MyVgg
from transform_net import ImageTransformNet
from util import *
from torch.utils.data import DataLoader
from torchvision import datasets


def train(style_image_fn, dataset_path, content_image_path, saving_path):
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    # constants
    BATCH_SIZE = 2
    IMAGE_SIZE = 256
    DATASET = dataset_path
    NUM_EPOCHS = 1
    CONTENT_WEIGHT = 1
    STYLE_WEIGHT = 1e6

    # data loader transformation
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor()])

    # set up data loader
    train_dataset = datasets.ImageFolder(DATASET, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # set up transformer net
    Net = ImageTransformNet().to(device)

    # set up Adam optimizer
    optimizer = Adam(Net.parameters(), lr=1e-4)

    # set up normalizer function
    normalize = normalizer(device)

    # import pre-trained VGG network
    vgg16 = MyVgg().to(device).eval()
    style_img = image_loader(style_image_fn, device)
    style_img = normalize(style_img)

    content_img = image_loader(content_image_path, device)

    # get fixed gram matrix of style features
    _, C, H, W = style_img.size()
    style_features = vgg16(style_img.repeat([BATCH_SIZE, 1, 1, 1]).to(device))
    gram_target_style_features = [gram(y) for y in style_features]

    l2_loss = torch.nn.MSELoss().to(device)

    def content_loss(content, target):
        """
        calculates the content loss, MSE loss
        :param content: input
        :param target: output of transformer net
        :return: MSE loss of content and target
        """
        return l2_loss(content, target)

    def style_loss(style, gram_target):
        """
        calculates total style loss
        :param style: style output of VGG net
        :param gram_target: gram matrix of target style features after VGG net
        :return: total style loss
        """
        total_style_loss = 0
        for i in range(len(gram_target)):
            total_style_loss += l2_loss(gram(style[i]), gram_target[i])
        return total_style_loss

    cp = 0
    for epoch in range(NUM_EPOCHS):
        print("========Epoch {}/{}========".format(epoch + 1, NUM_EPOCHS))
        batch_count = 0
        for y_c, _ in train_loader:
            optimizer.zero_grad()

            y_c = y_c.to(device)
            y_hat = Net(y_c.clone())

            y_hat = normalize(y_hat)
            y_c = normalize(y_c)

            y_hat_thru_net = vgg16(y_hat)
            y_c_thru_net = vgg16(y_c)

            loss_c = CONTENT_WEIGHT * content_loss(y_hat_thru_net[2], y_c_thru_net[2])

            loss_s = STYLE_WEIGHT * style_loss(y_hat_thru_net, gram_target_style_features)

            total_loss = loss_c + loss_s
            total_loss.backward()

            optimizer.step()

            if batch_count % 500 == 0:
                print(f'=======total loss is {total_loss}=========')
                print(f'=======current batch count is {batch_count}===========')
                print(f'=======style loss is {loss_s}==========')
                print(f'=======content loss is {loss_c}==========')

                # save model to path
                file_path = saving_path + f'epoch{epoch}_cp{cp}_iter{batch_count}.pth'
                cp += 1
                torch.save(Net.state_dict(), file_path)

                # evaluate current model
                Net.eval()
                with torch.no_grad():
                    content_img_tensor = Net(content_img)
                    show_image(content_img_tensor)
                    plt.show()
                Net.train()

            batch_count += 1

            if batch_count == 20001:
                break

    Net.eval()
    Net.cpu()

    file_path = 'network/' + 'transformer_weight.pth'
    torch.save(Net.state_dict(), file_path)
