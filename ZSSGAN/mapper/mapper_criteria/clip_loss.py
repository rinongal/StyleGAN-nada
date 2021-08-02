
import torch
import clip


class CLIPLoss(torch.nn.Module):

    def __init__(self, opts):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=opts.stylegan_size // 32)

        self.model_16, _ = clip.load("ViT-B/16", device="cuda")

        self.mse_loss = torch.nn.MSELoss()

    def forward(self, image, text):
        image = self.avg_pool(self.upsample(image))
        similarity = 1.0 * (1 - self.model(image, text)[0] / 100) + 1.0 * (1.0 - self.model_16(image, text)[0] / 100)
        return similarity

    def norm_loss(self, image_pre, image_post):
        norm_pre = self.model.encode_image(self.avg_pool(self.upsample(image_pre))).norm(dim=-1)
        norm_post = self.model.encode_image(self.avg_pool(self.upsample(image_post))).norm(dim=-1)

        return self.mse_loss(norm_pre, norm_post)